"""/v1/compress nets tool definitions a turn hook adds against its saving.

A hook that folds messages but adds a tool (skill search's search tool) sends
those tool tokens too; ``tokens_after`` must include them, as on the chat path.
"""

from __future__ import annotations

from typing import Any

from headroom.proxy.turn_hooks import register_turn_hook
from tests.gateway.conftest import compress

_LONG = "line of tool output that a hook can fold away entirely " * 60
_ADDED_TOOL = {
    "name": "search_skills",
    "description": "Find a skill by what it does. " * 40,
    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
}


class _FoldAndAddTool:
    name = savings_source = "skill_search"
    stream_safe = True

    def __init__(self, add_tool: bool) -> None:
        self.add_tool = add_tool

    def on_request(self, ctx: Any) -> None:
        ctx.messages = [{**m, "content": "folded"} for m in ctx.messages]
        if self.add_tool:
            ctx.tools = [*(ctx.tools or []), dict(_ADDED_TOOL)]


def _run(make_headroom_client: Any, *, add_tool: bool) -> dict[str, Any]:
    register_turn_hook(_FoldAndAddTool(add_tool))
    client = make_headroom_client(optimize=False)
    body = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": _LONG}],
        "tools": [{"name": "noop", "description": "x", "input_schema": {"type": "object"}}],
        "gateway": {"can_redrive": False, "can_relay_response": False},
    }
    resp = compress(client, body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_tools_a_hook_adds_count_as_sent(make_headroom_client) -> None:
    from headroom.proxy.turn_hooks import clear_turn_hooks

    plain = _run(make_headroom_client, add_tool=False)
    clear_turn_hooks()
    grown = _run(make_headroom_client, add_tool=True)
    assert plain["tokens_saved"] > 0
    added = grown["tokens_after"] - plain["tokens_after"]
    assert added > 100  # the search tool's definition
    assert grown["tokens_saved"] == plain["tokens_saved"] - added
    assert grown["tokens_before"] == plain["tokens_before"]
