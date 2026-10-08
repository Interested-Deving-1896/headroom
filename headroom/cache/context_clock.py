"""Conversation-scoped request clocks, separate from retrievable payloads."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ContextTurnSnapshot:
    current_turn: int
    # hash -> (compression event creation timestamp, original conversation turn)
    compression_turns: dict[str, tuple[float, int]]


@runtime_checkable
class ContextClockBackend(Protocol):
    """Optional capability; implementations atomically advance and retain age."""

    def observe_context_turn(
        self, conversation_key: str, hash_keys: Collection[str]
    ) -> ContextTurnSnapshot | None: ...


def context_conversation_key(
    session_id: str,
    workspace_key: str,
    messages: list[dict[str, Any]],
    *,
    explicit_session: bool,
) -> str:
    """Use client identity, or include the original conversation's first user turn.

    Model/system fallback IDs can serve independent agents in one workspace.
    The initial user message separates their origins without depending on the
    latest query, process-local tracker IDs, or optimized history. Byte-identical
    origins need explicit client session IDs to be distinguishable.
    """
    origin = None
    if not explicit_session:
        from .prefix_tracker import _canonicalize_for_prefix_compare

        first_user = next((message for message in messages if message.get("role") == "user"), None)
        if first_user is not None:
            origin = _canonicalize_for_prefix_compare([first_user])
    encoded = json.dumps([session_id, workspace_key, origin], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def advance_context_state(
    state: dict[str, Any] | None,
    events: Mapping[str, tuple[float, float]],
    live_hashes: Collection[str],
    now: float,
) -> tuple[dict[str, Any], ContextTurnSnapshot, float]:
    """Advance one owned-marker request; caller supplies its atomic boundary.

    Clock retention outlives every referenced compression event. An event's
    first turn survives missing markers, restarts, and cache eviction in the
    in-process ContextTracker. A new store creation timestamp resets only that
    event. Invalid durable state must fail closed, never start a fresh clock.
    """
    if state is None:
        state = {"version": 1, "turn": 0, "events": {}}
    if (
        not isinstance(state, dict)
        or state.get("version") != 1
        or type(state.get("turn")) is not int
        or state["turn"] < 0
        or not isinstance(state.get("events"), dict)
    ):
        raise ValueError("Invalid retained CCR conversation clock")
    anchors = {}
    for hash_key, anchor in state["events"].items():
        if (
            not isinstance(anchor, dict)
            or type(anchor.get("turn")) is not int
            or not 1 <= anchor["turn"] <= state["turn"]
            or not isinstance(anchor.get("created_at"), (int, float))
            or not isinstance(anchor.get("expires_at"), (int, float))
            or not math.isfinite(anchor["created_at"])
            or not math.isfinite(anchor["expires_at"])
        ):
            raise ValueError("Invalid retained CCR compression turn")
        if hash_key in live_hashes and anchor["expires_at"] >= now:
            anchors[hash_key] = anchor.copy()
    current_turn = state["turn"] + 1
    observed = {}
    for hash_key, (created_at, expires_at) in events.items():
        anchor = anchors.get(hash_key)
        if anchor is None or anchor["created_at"] != created_at:
            anchor = {"created_at": created_at, "turn": current_turn, "expires_at": expires_at}
            anchors[hash_key] = anchor
        observed[hash_key] = (created_at, anchor["turn"])
    retained = {"version": 1, "turn": current_turn, "events": anchors}
    expires_at = max(anchor["expires_at"] for anchor in anchors.values())
    return retained, ContextTurnSnapshot(current_turn, observed), expires_at
