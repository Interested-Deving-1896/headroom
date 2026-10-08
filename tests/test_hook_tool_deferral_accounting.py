"""A turn hook's measured tool delta ignores the ``defer_loading`` flag.

Deferral is booked through ``tool_search_deferred_tokens``. Measuring the raw
tool array would see only the flag text appear or disappear, so un-deferring a
tool (tool search's hot-tools hook) was booked as a realized saving although
it makes the request larger.
"""

from __future__ import annotations

import json

import pytest

from headroom.proxy.savings_attribution import from_tags
from headroom.proxy.tool_schema_savings_policy import without_deferral_flags
from headroom.proxy.turn_hooks import (
    TurnContext,
    clear_turn_hooks,
    register_turn_hook,
    run_request_hooks,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_turn_hooks()
    yield
    clear_turn_hooks()


def _tool(name: str, size: int = 400) -> dict:
    return {"name": name, "description": "d" * size, "input_schema": {"type": "object"}}


def _count(value: object) -> int:
    # Stand-in tokenizer: one token per 4 characters of the serialised array.
    return len(json.dumps(value, default=str)) // 4 if value else 0


class _UndeferHook:
    """Shaped like headroom-tool-search's HotToolsHook: drops the flag in place."""

    name = savings_source = "tool_search"
    stream_safe = True

    def on_request(self, ctx: TurnContext) -> None:
        for tool in ctx.tools:
            tool.pop("defer_loading", None)


class _DeferHook:
    """Shaped like NativeDeferralHook: flags tools, books through the tag."""

    name = savings_source = "tool_search"
    stream_safe = True

    def on_request(self, ctx: TurnContext) -> None:
        ctx.tools = [{**t, "defer_loading": True} for t in ctx.tools]


class _RemoveHook:
    """Shaped like a router: really removes a tool from the array."""

    name = savings_source = "tool_router"
    stream_safe = True

    def on_request(self, ctx: TurnContext) -> None:
        ctx.tools = ctx.tools[:1]


def _run(hook: object, tools: list) -> list[dict]:
    register_turn_hook(hook)
    ctx = TurnContext(
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        count_messages=_count,
        count_tools=_count,
    )
    run_request_hooks(ctx)
    return from_tags(ctx.tags)


def test_undeferring_a_tool_is_not_booked_as_a_saving() -> None:
    tools = [{**_tool("a"), "defer_loading": True}, {**_tool("b"), "defer_loading": True}]
    assert _run(_UndeferHook(), tools) == []


def test_deferring_is_left_to_the_deferral_tag() -> None:
    # Only the flag changes, so the measured delta is zero and the hook's own
    # tool_search_deferred_tokens tag stays the single place deferral is booked.
    assert _run(_DeferHook(), [_tool("a"), _tool("b")]) == []


def test_a_real_removal_is_still_measured() -> None:
    (entry,) = _run(_RemoveHook(), [_tool("a"), _tool("b")])
    assert entry["source"] == "tool_router"
    assert entry["tokens"] == _count([_tool("a"), _tool("b")]) - _count([_tool("a")])


def test_without_deferral_flags_never_mutates_and_passes_non_lists() -> None:
    tools = [{**_tool("a"), "defer_loading": True}, "not-a-dict"]
    out = without_deferral_flags(tools)
    assert "defer_loading" in tools[0]
    assert out[0] == _tool("a")
    assert out[1] == "not-a-dict"
    assert without_deferral_flags(None) is None
