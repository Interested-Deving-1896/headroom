"""Live OpenAI API tests for the Codex (role="tool" message) path.

Same contract as the Anthropic live tests: the real API must accept our
transformed message shapes, and the model must answer correctly from a
deduped conversation.

Skipped without OPENAI_API_KEY. Costs: a few hundred gpt-4o-mini tokens/run.
"""

from __future__ import annotations

import json
import os

import httpx
import pytest

from headroom.config import ReadLifecycleConfig
from headroom.transforms.read_lifecycle import ReadLifecycleManager

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

MODEL = "gpt-4o-mini"
API_URL = "https://api.openai.com/v1/chat/completions"

FILE_CONTENT = (
    '     1\tdef answer():\n     2\t    """Returns the magic number."""\n     3\t    return 42\n'
) + "".join(f"    {i}\t# padding line {i}\n" for i in range(4, 40))

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "Read",
        "description": "Read a file",
        "parameters": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
}


def call_openai(messages: list[dict]) -> str:
    resp = httpx.post(
        API_URL,
        json={
            "model": MODEL,
            "max_tokens": 150,
            "tools": [READ_TOOL],
            "messages": messages,
        },
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        timeout=60,
    )
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:500]}"
    return resp.json()["choices"][0]["message"]["content"] or ""


def read_roundtrip(tc_id: str, content: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "/src/magic.py"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": tc_id, "content": content},
    ]


class TestDedupMarkerLiveOpenAI:
    def test_model_answers_from_earlier_copy_via_pointer(self):
        messages = [{"role": "user", "content": "Read /src/magic.py please"}]
        messages += read_roundtrip("call_r1", FILE_CONTENT)
        messages += [
            {"role": "assistant", "content": "Read it. Anything else?"},
            {"role": "user", "content": "Read it again to double-check."},
        ]
        messages += read_roundtrip("call_r2", FILE_CONTENT)

        res = ReadLifecycleManager(ReadLifecycleConfig()).apply(messages)
        assert res.reads_dedup_repeat == 1, "fixture must trigger dedup"

        res.messages.append(
            {
                "role": "user",
                "content": "Based on the file contents: what number does answer() return? "
                "Reply with just the number.",
            }
        )
        reply = call_openai(res.messages)
        assert "42" in reply, f"model failed to answer from deduped context: {reply!r}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
