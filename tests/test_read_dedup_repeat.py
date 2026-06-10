"""Tests for first-sight repeat-Read dedup (ReadState.DEDUP_REPEAT).

Covers both client wire formats:
- Anthropic (Claude Code): tool_use / tool_result content blocks
- OpenAI (Codex and friends): assistant tool_calls + role="tool" messages

The mechanism's safety contract, each clause tested below:
1. Only the NEW copy is replaced; the earlier copy keeps verbatim bytes.
2. Only byte-identical content dedups (post-edit re-reads never match).
3. The frozen prefix is never mutated.
4. Replacement is deterministic (byte-identical across passes) so the
   compression cache replays it stably across turns.
5. Stale classification wins over dedup (a stale read is factually wrong;
   pointing at it would be worse than replacing it).
"""

from __future__ import annotations

import pytest

from headroom.config import ReadLifecycleConfig
from headroom.transforms.read_lifecycle import ReadLifecycleManager, ReadState

# > min_size_bytes (512) so replacement is not skipped as tiny.
CONTENT = "     1\tdef foo():\n     2\t    return 42\n" * 30
CONTENT_EDITED = CONTENT + "     3\tCHANGED\n"


def make_manager(**overrides) -> ReadLifecycleManager:
    return ReadLifecycleManager(ReadLifecycleConfig(**overrides))


# ─── Message builders ────────────────────────────────────────────────────


def anthropic_read(tc_id: str, file_path: str, content: str) -> list[dict]:
    """One Read round-trip in Anthropic content-block format."""
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": tc_id, "name": "Read", "input": {"file_path": file_path}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tc_id, "content": content}],
        },
    ]


def anthropic_edit(tc_id: str, file_path: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tc_id,
                    "name": "Edit",
                    "input": {"file_path": file_path, "old_string": "a", "new_string": "b"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tc_id, "content": "ok"}],
        },
    ]


def openai_read(tc_id: str, file_path: str, content: str) -> list[dict]:
    """One Read round-trip in OpenAI tool-call format (Codex)."""
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc_id,
                    "function": {"name": "Read", "arguments": f'{{"file_path": "{file_path}"}}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": tc_id, "content": content},
    ]


def anthropic_repeat_conv() -> list[dict]:
    msgs = [{"role": "user", "content": "look at foo.py"}]
    msgs += anthropic_read("r1", "/x/foo.py", CONTENT)
    msgs += anthropic_read("r2", "/x/foo.py", CONTENT)
    return msgs


# ─── Anthropic format ────────────────────────────────────────────────────


class TestAnthropicFormat:
    def test_repeat_read_dedups_new_copy_only(self):
        res = make_manager().apply(anthropic_repeat_conv())

        assert res.reads_dedup_repeat == 1
        assert res.reads_total == 2
        # Earlier copy: verbatim, untouched.
        assert res.messages[2]["content"][0]["content"] == CONTENT
        # New copy: pointer marker with the load-bearing retrieval phrase.
        marker = res.messages[4]["content"][0]["content"]
        assert "byte-identical" in marker
        assert "/x/foo.py" in marker
        assert "Retrieve original: hash=" in marker

    def test_differing_content_not_deduped(self):
        msgs = [{"role": "user", "content": "look"}]
        msgs += anthropic_read("r1", "/x/foo.py", CONTENT)
        msgs += anthropic_read("r2", "/x/foo.py", CONTENT_EDITED)
        res = make_manager().apply(msgs)

        assert res.reads_dedup_repeat == 0
        assert res.messages[4]["content"][0]["content"] == CONTENT_EDITED

    def test_different_files_not_deduped(self):
        msgs = [{"role": "user", "content": "look"}]
        msgs += anthropic_read("r1", "/x/foo.py", CONTENT)
        msgs += anthropic_read("r2", "/x/bar.py", CONTENT)
        res = make_manager().apply(msgs)

        assert res.reads_dedup_repeat == 0

    def test_three_identical_reads_point_at_first(self):
        msgs = [{"role": "user", "content": "look"}]
        msgs += anthropic_read("r1", "/x/foo.py", CONTENT)
        msgs += anthropic_read("r2", "/x/foo.py", CONTENT)
        msgs += anthropic_read("r3", "/x/foo.py", CONTENT)
        res = make_manager().apply(msgs)

        assert res.reads_dedup_repeat == 2
        # First copy survives verbatim; both repeats are markers.
        assert res.messages[2]["content"][0]["content"] == CONTENT
        assert "byte-identical" in res.messages[4]["content"][0]["content"]
        assert "byte-identical" in res.messages[6]["content"][0]["content"]

    def test_stale_wins_over_dedup(self):
        # read, read (identical), then Edit: both reads are stale — the
        # edit invalidated them. Neither should be a dedup pointer.
        msgs = [{"role": "user", "content": "look"}]
        msgs += anthropic_read("r1", "/x/foo.py", CONTENT)
        msgs += anthropic_read("r2", "/x/foo.py", CONTENT)
        msgs += anthropic_edit("e1", "/x/foo.py")
        res = make_manager().apply(msgs)

        assert res.reads_stale == 2
        assert res.reads_dedup_repeat == 0
        assert "stale" in res.messages[2]["content"][0]["content"]
        assert "stale" in res.messages[4]["content"][0]["content"]

    def test_frozen_prefix_never_mutated(self):
        msgs = anthropic_repeat_conv()
        res = make_manager().apply(msgs, frozen_message_count=len(msgs))

        assert res.reads_dedup_repeat == 0
        assert res.messages[4]["content"][0]["content"] == CONTENT

    def test_dedup_of_live_copy_against_frozen_original_allowed(self):
        # The earlier copy being inside the frozen prefix is fine — it is
        # not modified; only the live-zone repeat becomes a pointer.
        msgs = anthropic_repeat_conv()
        res = make_manager().apply(msgs, frozen_message_count=3)

        assert res.reads_dedup_repeat == 1
        assert res.messages[2]["content"][0]["content"] == CONTENT
        assert "byte-identical" in res.messages[4]["content"][0]["content"]

    def test_tiny_content_skipped(self):
        small = "     1\tok\n"
        msgs = [{"role": "user", "content": "look"}]
        msgs += anthropic_read("r1", "/x/foo.py", small)
        msgs += anthropic_read("r2", "/x/foo.py", small)
        res = make_manager().apply(msgs)

        assert res.reads_dedup_repeat == 0
        assert res.messages[4]["content"][0]["content"] == small

    def test_deterministic_across_passes(self):
        m = make_manager()
        first = m.apply(anthropic_repeat_conv()).messages[4]["content"][0]["content"]
        second = m.apply(anthropic_repeat_conv()).messages[4]["content"][0]["content"]
        assert first == second

    def test_flag_off_disables_dedup(self):
        res = make_manager(dedup_repeat_reads=False).apply(anthropic_repeat_conv())
        assert res.reads_dedup_repeat == 0
        assert res.messages[4]["content"][0]["content"] == CONTENT


# ─── OpenAI / Codex format ──────────────────────────────────────────────


class TestOpenAIFormat:
    def _conv(self) -> list[dict]:
        msgs = [{"role": "user", "content": "look at foo.py"}]
        msgs += openai_read("r1", "/x/foo.py", CONTENT)
        msgs += openai_read("r2", "/x/foo.py", CONTENT)
        return msgs

    def test_repeat_read_dedups_new_copy_only(self):
        res = make_manager().apply(self._conv())

        assert res.reads_dedup_repeat == 1
        assert res.messages[2]["content"] == CONTENT
        marker = res.messages[4]["content"]
        assert "byte-identical" in marker
        assert "Retrieve original: hash=" in marker

    def test_differing_content_not_deduped(self):
        msgs = [{"role": "user", "content": "look"}]
        msgs += openai_read("r1", "/x/foo.py", CONTENT)
        msgs += openai_read("r2", "/x/foo.py", CONTENT_EDITED)
        res = make_manager().apply(msgs)

        assert res.reads_dedup_repeat == 0

    def test_mixed_formats_in_one_conversation(self):
        # Anthropic-format read first, OpenAI-format repeat after — the
        # content map is format-agnostic, so the repeat still dedups.
        msgs = [{"role": "user", "content": "look"}]
        msgs += anthropic_read("r1", "/x/foo.py", CONTENT)
        msgs += openai_read("r2", "/x/foo.py", CONTENT)
        res = make_manager().apply(msgs)

        assert res.reads_dedup_repeat == 1
        assert res.messages[2]["content"][0]["content"] == CONTENT
        assert "byte-identical" in res.messages[4]["content"]


# ─── CCR recovery path ──────────────────────────────────────────────────


class TestCcrIntegration:
    def test_original_stored_and_retrievable(self):
        from headroom.cache.backends.memory import InMemoryBackend
        from headroom.cache.compression_store import CompressionStore

        store = CompressionStore(backend=InMemoryBackend())
        mgr = ReadLifecycleManager(ReadLifecycleConfig(), compression_store=store)
        res = mgr.apply(anthropic_repeat_conv())

        assert res.reads_dedup_repeat == 1
        assert len(res.ccr_hashes) == 1
        entry = store.retrieve(res.ccr_hashes[0])
        assert entry is not None
        assert entry.original_content == CONTENT
        assert entry.compression_strategy == f"read_lifecycle:{ReadState.DEDUP_REPEAT.value}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
