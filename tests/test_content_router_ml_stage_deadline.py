"""Request-scoped ML ceiling for the mixed-content path (#3711).

``HEADROOM_KOMPRESS_MAX_TOKENS`` bounds a single block. It cannot bound a
request: ``_compress_mixed`` splits a payload into sections and calls
``_try_ml_compressor`` once per section, so every section can sit under the
per-block ceiling while their sum runs for minutes. The reporter measured ~70s
on a 1.4MB ``tool_result`` of prose blocks separated by small JSON objects,
which blew the 30s compression budget and then quarantined compression for
every following request -- while
``headroom_kompress_size_gate_total`` recorded only ``within``.

A stub stands in for Kompress so these assert the *budget*, not ONNX latency:
the real model is not available in CI, and the bug is about how many times a
slow stage is entered, not how slow it is.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass

import pytest

from headroom.transforms import content_router as cr


class _Tokenizer:
    def count_text(self, content: str) -> int:
        return len(content.split())


@dataclass
class _StubResult:
    compressed: str
    compressed_tokens: int


class _SlowKompress:
    """Stands in for Kompress: ready, and costs `per_call_s` every call."""

    def __init__(self, per_call_s: float) -> None:
        self.per_call_s = per_call_s
        self.calls = 0

    def is_ready(self) -> bool:
        return True

    def ensure_background_load(self) -> None:  # pragma: no cover - never reached
        raise AssertionError("stub is always ready")

    def compress(self, text: str, **_kwargs: object) -> _StubResult:
        self.calls += 1
        time.sleep(self.per_call_s)
        out = text[: max(1, len(text) // 2)]
        return _StubResult(compressed=out, compressed_tokens=cr._estimate_tokens(out))


def _prose(n_chars: int, seed: int = 7) -> str:
    rng = random.Random(seed)
    words = (
        "analysis deployment configuration throughput latency pipeline compression "
        "artifact identifier resolution boundary telemetry inference workspace"
    ).split()
    out: list[str] = []
    total = 0
    while total < n_chars:
        line = " ".join(rng.choice(words) for _ in range(14))
        out.append(line)
        total += len(line) + 1
    return "\n".join(out)


def _reporter_payload(block_chars: int = 40_000, blocks: int = 8) -> str:
    """Prose blocks separated by small JSON objects -- the #3711 shape.

    Each block stays under the 50k-token per-block gate, so the size gate
    cannot fire; only a request-scoped ceiling can stop this.
    """
    parts: list[str] = []
    for i in range(blocks):
        parts.append(_prose(block_chars, seed=i))
        parts.append(json.dumps({"id": i, "status": "ok", "note": f"marker {i}"}))
    return "\n\n".join(parts)


def _tool_result_messages() -> list[dict]:
    """The reported shape: the payload arrives as a tool_result, not user text.

    Plain user text is protected from compression
    (``router:protected:user_message``) and never reaches the ML stage at all.
    """
    return [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": _reporter_payload()}
            ],
        },
    ]


@pytest.fixture
def gate_outcomes(monkeypatch) -> dict[str, int]:
    seen: dict[str, int] = {}
    monkeypatch.setattr(
        cr.ContentRouter,
        "_observe_kompress_size_gate",
        lambda self, outcome: seen.__setitem__(outcome, seen.get(outcome, 0) + 1),
    )
    return seen


@pytest.fixture
def slow_kompress(monkeypatch) -> _SlowKompress:
    stub = _SlowKompress(per_call_s=0.05)
    monkeypatch.setattr(cr.ContentRouter, "_get_kompress", lambda self: stub)
    return stub


def _apply(router: cr.ContentRouter) -> None:
    router.apply(
        _tool_result_messages(),
        _Tokenizer(),
        frozen_message_count=1,
        min_tokens_to_compress=1,
    )


def test_size_gate_alone_cannot_bound_a_request(gate_outcomes, slow_kompress, monkeypatch) -> None:
    """Every section passes the per-block gate -- this is the #3711 precondition.

    Reproduces the reporter's metric exactly: many ``within`` decisions and no
    ``exceeded``. If this ever records ``exceeded`` the payload has stopped
    reproducing the report, and the deadline test below would pass for the
    wrong reason.
    """
    monkeypatch.setenv("HEADROOM_ML_STAGE_DEADLINE_SECONDS", "0")  # isolate the size gate

    _apply(cr.ContentRouter(cr.ContentRouterConfig()))

    assert gate_outcomes.get("exceeded", 0) == 0, (
        f"payload no longer reproduces #3711 (a section exceeded the gate): {gate_outcomes}"
    )
    assert gate_outcomes.get("within", 0) > 1, (
        "expected repeated per-section ML entry, the shape that blows the budget; "
        f"got {gate_outcomes}"
    )
    assert slow_kompress.calls > 1, "the ML stage was entered once or not at all"


def test_ml_deadline_stops_further_ml_work_once_the_budget_is_spent(
    gate_outcomes, slow_kompress, monkeypatch
) -> None:
    """Past the ceiling, remaining sections route off ML instead of compounding."""
    monkeypatch.setenv("HEADROOM_ML_STAGE_DEADLINE_SECONDS", "0.06")

    _apply(cr.ContentRouter(cr.ContentRouterConfig()))

    assert gate_outcomes.get("deadline", 0) > 0, (
        f"request-scoped ceiling never fired: {gate_outcomes}"
    )

    # The saving has to be real, so measure it: the same payload with the
    # ceiling disabled must enter the model strictly more often.
    bounded_calls = slow_kompress.calls
    monkeypatch.setenv("HEADROOM_ML_STAGE_DEADLINE_SECONDS", "0")
    unbounded = _SlowKompress(per_call_s=0.05)
    monkeypatch.setattr(cr.ContentRouter, "_get_kompress", lambda self: unbounded)

    _apply(cr.ContentRouter(cr.ContentRouterConfig()))

    assert bounded_calls < unbounded.calls, (
        f"ceiling skipped no model calls: {bounded_calls} with it, {unbounded.calls} without"
    )


def test_deadline_zero_restores_previous_unbounded_behaviour(
    gate_outcomes, slow_kompress, monkeypatch
) -> None:
    monkeypatch.setenv("HEADROOM_ML_STAGE_DEADLINE_SECONDS", "0")

    _apply(cr.ContentRouter(cr.ContentRouterConfig()))

    assert gate_outcomes.get("deadline", 0) == 0, (
        f"deadline fired although disabled: {gate_outcomes}"
    )


def test_direct_compress_callers_stay_unarmed() -> None:
    """``compress()`` without ``apply()`` keeps the old unbounded behaviour.

    Tests and the ``/v1/compress`` path call ``compress()`` directly; they must
    not inherit a request budget nobody set.
    """
    router = cr.ContentRouter(cr.ContentRouterConfig())
    assert router._runtime_state_var.get().ml_deadline is None
