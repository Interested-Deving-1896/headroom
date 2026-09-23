"""Tests for cache_control breakpoint diagnostics and log-privacy switches.

Covers the three pieces added for the uncached-tail investigation:
- ``count_cache_breakpoints`` / ``log_cache_breakpoints`` (proxy helpers)
- the ``HEADROOM_LOG_PAYLOAD_PREVIEW`` kill switch (compression store)
- the injection guard that keeps proactive expansion out of breakpointed blocks
"""

from __future__ import annotations

import logging
import stat
from contextlib import contextmanager
from pathlib import Path

import pytest

from headroom.cache.compression_store import _payload_for_retrieval_log
from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.helpers import count_cache_breakpoints, log_cache_breakpoints

_CC = {"cache_control": {"type": "ephemeral"}}


def _claude_code_style_request() -> tuple[list[dict], list[dict], list[dict]]:
    """System/messages/tools shaped like a real Claude Code request."""
    system = [
        {"type": "text", "text": "You are Claude Code."},
        {"type": "text", "text": "project instructions", **_CC},
    ]
    tools = [
        {"name": "Bash", "input_schema": {}},
        {"name": "Read", "input_schema": {}, **_CC},
    ]
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi", **_CC}]},
        {"role": "assistant", "content": [{"type": "text", "text": "ack"}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "big output"}],
                    **_CC,
                }
            ],
        },
    ]
    return system, messages, tools


def test_count_cache_breakpoints_counts_all_sections() -> None:
    system, messages, tools = _claude_code_style_request()
    stats = count_cache_breakpoints(system, messages, tools)
    assert stats["system"] == 1
    assert stats["tools"] == 1
    assert stats["messages"] == 2
    assert stats["total"] == 4
    assert stats["message_count"] == 3
    assert stats["last_marker_tail"] == 0  # last message carries a marker


def test_count_cache_breakpoints_counts_nested_tool_result_markers() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "out", **_CC}],
                }
            ],
        }
    ]
    stats = count_cache_breakpoints("plain system string", messages, None)
    assert stats["system"] == 0
    assert stats["tools"] == 0
    assert stats["messages"] == 1
    assert stats["last_marker_tail"] == 0


def test_count_cache_breakpoints_tail_tracks_last_marker() -> None:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "a", **_CC}]},
        {"role": "assistant", "content": [{"type": "text", "text": "b"}]},
        {"role": "user", "content": [{"type": "text", "text": "c"}]},
    ]
    stats = count_cache_breakpoints(None, messages, None)
    assert stats["last_marker_tail"] == 2
    assert count_cache_breakpoints(None, [], None)["last_marker_tail"] == -1


def test_log_cache_breakpoints_warns_on_dropped_marker(caplog) -> None:
    system, messages, tools = _claude_code_style_request()
    inbound = count_cache_breakpoints(system, messages, tools)
    # Transform "lost" the final breakpoint: strip it from the last message.
    stripped = [dict(m) for m in messages]
    stripped[2] = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "compressed"}],
    }
    outbound = count_cache_breakpoints(system, stripped, tools)
    with caplog.at_level(logging.INFO, logger="headroom.proxy"):
        log_cache_breakpoints(request_id="r1", inbound=inbound, outbound=outbound)
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert "dropped=true" in record.getMessage()
    assert "tail_grew=true" in record.getMessage()


def test_log_cache_breakpoints_info_when_preserved(caplog) -> None:
    system, messages, tools = _claude_code_style_request()
    stats = count_cache_breakpoints(system, messages, tools)
    with caplog.at_level(logging.INFO, logger="headroom.proxy"):
        log_cache_breakpoints(request_id="r1", inbound=stats, outbound=stats)
    [record] = caplog.records
    assert record.levelno == logging.INFO
    assert "dropped=false" in record.getMessage()


def test_payload_preview_disabled_omits_content(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_LOG_PAYLOAD_PREVIEW", "0")
    payload = "secret file contents: api_key=sk-abcdefghijklmnop"
    event = _payload_for_retrieval_log(payload)
    assert event["payload_preview"] == ""
    assert event["payload_preview_chars"] == 0
    assert event["payload_chars"] == len(payload)
    assert event["payload_truncated"] is True


def test_payload_preview_disabled_by_default(monkeypatch) -> None:
    """Unset means off: the log gets byte counts, never the content."""
    monkeypatch.delenv("HEADROOM_LOG_PAYLOAD_PREVIEW", raising=False)
    event = _payload_for_retrieval_log("hello world")
    assert event["payload_preview"] == ""
    assert event["payload_preview_chars"] == 0
    assert event["payload_chars"] == len("hello world")


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_payload_preview_opt_in_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv("HEADROOM_LOG_PAYLOAD_PREVIEW", value)
    assert _payload_for_retrieval_log("hello world")["payload_preview"] == "hello world"


@pytest.mark.parametrize("value", ["", "0", "off", "no", "maybe", "  "])
def test_payload_preview_stays_off_for_anything_else(monkeypatch, value: str) -> None:
    """Only an explicit opt-in turns previews on — a typo must not."""
    monkeypatch.setenv("HEADROOM_LOG_PAYLOAD_PREVIEW", value)
    assert _payload_for_retrieval_log("hello world")["payload_preview"] == ""


def test_append_context_skips_breakpointed_text_block() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "breakpointed", **_CC},
                {"type": "text", "text": "free"},
            ],
        }
    ]
    result = AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
        messages, "CTX", frozen_message_count=0
    )
    blocks = result[0]["content"]
    assert blocks[0]["text"] == "breakpointed"  # untouched
    assert blocks[1]["text"].endswith("CTX")


def test_append_context_no_eligible_block_returns_unchanged() -> None:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "breakpointed", **_CC}],
        }
    ]
    result = AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
        messages, "CTX", frozen_message_count=0
    )
    assert result == messages


def test_count_cache_breakpoints_tolerates_malformed_shapes() -> None:
    messages = [
        "not-a-dict",
        {"role": "user", "content": ["scalar-block", {"type": "text", "text": "x", **_CC}]},
        {"role": "user", "content": "plain string"},
    ]
    stats = count_cache_breakpoints("system-as-string", messages, "tools-as-string")
    assert stats["system"] == 0
    assert stats["tools"] == 0
    assert stats["messages"] == 1
    assert stats["message_count"] == 3
    assert stats["last_marker_tail"] == 1

    empty = count_cache_breakpoints(None, None, None)
    assert empty["total"] == 0
    assert empty["message_count"] == 0


# --- the runtime log file itself -------------------------------------------
#
# _payload_for_retrieval_log decides what goes into the record;
# _setup_file_logging decides who can read the file it lands in. Both halves
# of the default-off guarantee are checked against a real log on disk.


@contextmanager
def _proxy_log(tmp_path, monkeypatch, port: int):
    """Point the workspace at *tmp_path*, install the real proxy log handler."""
    from headroom.proxy.helpers import _PROXY_LOG_HANDLER_NAME, _setup_file_logging

    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    headroom_logger = logging.getLogger("headroom")
    before = list(headroom_logger.handlers)
    propagate = headroom_logger.propagate
    try:
        _setup_file_logging(port)
        [handler] = [h for h in headroom_logger.handlers if h.name == _PROXY_LOG_HANDLER_NAME]
        yield Path(handler.baseFilename)
    finally:
        for handler in list(headroom_logger.handlers):
            if handler not in before:
                headroom_logger.removeHandler(handler)
                handler.close()
        headroom_logger.propagate = propagate


def test_runtime_log_holds_no_payload_text_at_default_settings(tmp_path, monkeypatch) -> None:
    """A retrieval on default settings leaves byte counts in the log, not content."""
    from headroom.cache.compression_store import CompressionStore

    monkeypatch.delenv("HEADROOM_LOG_PAYLOAD_PREVIEW", raising=False)
    secret = "BEGIN-CUSTOMER-DATA ssn=123-45-6789 def sekrit(): pass END-CUSTOMER-DATA"

    with _proxy_log(tmp_path, monkeypatch, 18801) as log_path:
        store = CompressionStore(enable_feedback=False)
        assert store.retrieve(store.store(original=secret, compressed="[compressed]")) is not None
        logging.getLogger("headroom").handlers[-1].flush()
        text = log_path.read_text(encoding="utf-8")

    assert "event=headroom_retrieve" in text, "the retrieval was not logged at all"
    assert secret not in text
    assert "123-45-6789" not in text
    assert f'"payload_chars":{len(secret)}' in text
    assert '"payload_preview":""' in text


def test_runtime_log_is_owner_only_when_preview_enabled(tmp_path, monkeypatch) -> None:
    """Opting in to previews hardens the log the previews land in."""
    monkeypatch.setenv("HEADROOM_LOG_PAYLOAD_PREVIEW", "1")
    with _proxy_log(tmp_path, monkeypatch, 18802) as log_path:
        assert log_path.exists()
        assert stat.S_IMODE(log_path.stat().st_mode) == 0o600


def test_runtime_log_hardening_survives_a_pre_existing_world_readable_log(
    tmp_path, monkeypatch
) -> None:
    """O_CREAT's mode does not apply to an existing file; the chmod must."""
    monkeypatch.setenv("HEADROOM_LOG_PAYLOAD_PREVIEW", "1")
    from headroom import paths as _paths

    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    stale = _paths.proxy_log_path(18803)
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("from an older, unhardened run\n", encoding="utf-8")
    stale.chmod(0o644)

    with _proxy_log(tmp_path, monkeypatch, 18803) as log_path:
        assert stat.S_IMODE(log_path.stat().st_mode) == 0o600
