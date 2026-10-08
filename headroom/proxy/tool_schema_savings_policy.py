"""Tool-schema savings attribution policy for proxy stats.

Headroom saves input tokens in two accounting shapes, and the split is not a
style choice — it follows from what each transform can observe:

* **Compaction** (``*:tool_schema_compaction``, ``*:tool_desc_compaction``)
  rewrites the tool array in place, so both endpoints are countable. Handlers
  fold the delta into ``original_tokens``/``optimized_tokens``, which keeps
  ``tok_before - tok_after == tok_saved`` coherent in the PERF line. It is
  therefore ALREADY inside ``tokens_saved`` and must never be added again.
* **Deferral / hook shrink** (tool search, turn hooks) removes schemas that
  ``count_messages`` never saw, so it cannot move ``original_tokens``. It is
  recorded in per-request tags and is ADDITIVE to ``tokens_saved``.

The one rule a caller needs: the headline is
``tokens_saved + tool_schema_saved_from_tags(tags)``. Use
:func:`headline_tokens_saved` rather than open-coding it — three surfaces had
drifted inline copies of that sum, and two harnesses were silently dropping
their compaction savings entirely because the convention was never written down.

Adding a new tool-schema-shrinking feature? If it moves the tool array, fold it
in the handler like the compaction sites do. If it defers schemas, add its tag
name to :data:`TOOL_SCHEMA_SAVINGS_TAGS` and every surface picks it up.
"""

from __future__ import annotations

import copy
from typing import Any


def without_deferral_flags(tools: object) -> object:
    """The tool array as it is MEASURED for a hook's tool delta.

    Deferral is reported through ``tool_search_deferred_tokens`` (it removes
    schemas from the billed context while leaving them in the array, flagged
    ``defer_loading``). Measuring the raw array would see only the flag
    appear or disappear: un-deferring a tool would shorten the JSON by the
    flag text and be booked as a small saving although the bill goes UP, and
    deferring one would add a phantom cost. So the flag is ignored here, and a
    hook's measured tool delta covers only real removals and rewrites. Never
    mutates its input.
    """
    if not isinstance(tools, list):
        return tools
    return [
        {k: v for k, v in tool.items() if k != "defer_loading"}
        if isinstance(tool, dict) and "defer_loading" in tool
        else tool
        for tool in tools
    ]


def deferred_booking(tags: object, tools: object) -> tuple[int, dict[str, Any]] | None:
    """Snapshot the deferral currently booked: the tag and the tools it covers.

    ``None`` when nothing is booked. The schemas are copied as they were when
    booked, so a later in-place edit cannot change what the credit stands for.
    """
    if not isinstance(tags, dict) or "tool_search_deferred_tokens" not in tags:
        return None
    try:
        booked = int(tags.get("tool_search_deferred_tokens") or 0)
    except (TypeError, ValueError):
        return None
    snapshot = {
        str(t["name"]): copy.deepcopy(t)
        for t in (tools if isinstance(tools, list) else [])
        if isinstance(t, dict) and t.get("defer_loading") and t.get("name")
    }
    return booked, snapshot


def reconcile_deferred_tokens(
    tags: object, booking: tuple[int, dict[str, Any]] | None, tools: object, count_tools: object
) -> None:
    """Cap the deferral credit at the booked tools still deferred after hooks.

    A hook may un-defer tools the deferral booked (tool search's hot tools);
    those are billed, so their share of the credit goes. The remaining share is
    recounted from the BOOKED schemas, so a hook that grows another deferred
    schema cannot keep an un-deferred tool's credit alive. Only ever lowers the
    tag, and lowers the matching ``tool_search`` ledger entry by the same
    amount so ``/stats`` by-source agrees with the headline. Never raises.
    """
    if booking is None or not isinstance(tags, dict) or not callable(count_tools):
        return
    booked, snapshot = booking
    try:
        still = {
            str(t["name"])
            for t in (tools if isinstance(tools, list) else [])
            if isinstance(t, dict) and t.get("defer_loading") and t.get("name")
        }
        kept = [schema for name, schema in snapshot.items() if name in still]
        measured = int(count_tools(kept)) if kept else 0
        released = booked - measured
        if released <= 0:
            return
        tags["tool_search_deferred_tokens"] = measured
        tags["tool_search_deferred_tools"] = len(kept)
        from headroom.proxy.savings_attribution import SAVINGS_ATTRIBUTION_TAG

        ledger = tags.get(SAVINGS_ATTRIBUTION_TAG)
        for item in reversed(ledger if isinstance(ledger, list) else []):
            if isinstance(item, dict) and item.get("source") == "tool_search":
                item["tokens"] = max(0, int(item.get("tokens") or 0) - released)
                break
    except Exception:  # accounting must never break a request
        return


TOOL_SCHEMA_SAVINGS_TAGS: tuple[str, ...] = (
    "tool_search_deferred_tokens",
    "turn_hook_tools_saved_tokens",
)


def tool_schema_saved_from_tags(tags: object) -> int:
    """Return tool-definition tokens Headroom kept out of context for one request.

    The summed tags are set only on paths where Headroom performed the deferral,
    so clients that already had tool search enabled contribute zero here.

    These tags are additive to ``tokens_saved`` — see the module docstring.
    """
    if not isinstance(tags, dict):
        return 0

    total = 0
    for key in TOOL_SCHEMA_SAVINGS_TAGS:
        try:
            total += int(tags.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
    return total


def headline_tokens_saved(tokens_saved: object, tags: object) -> int:
    """Return the single "Tokens saved" figure for one request.

    This is the only correct total to show a user: message compression plus the
    tool-definition tokens that never entered the context. Every reporting
    surface (PERF line, ``headroom perf``, ``/api/stats``, dashboard, session
    summary) must route through here so they cannot disagree.

    Clamped at zero: handlers already revert any inflation before forwarding, so
    a negative is a token-count artifact that never reached the model.
    """
    base = 0
    if isinstance(tokens_saved, (int, float, str)) and not isinstance(tokens_saved, bool):
        try:
            base = int(tokens_saved)
        except (TypeError, ValueError):
            base = 0
    return max(0, base + tool_schema_saved_from_tags(tags))
