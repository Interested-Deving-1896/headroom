"""fix-3: the cache-mode delta path must ignore moved cache_control markers.

Cache mode replays the exact previously-forwarded bytes for history and
compresses ONLY the newly appended delta (compress-once-then-freeze). The gate
is ``AnthropicHandler._extract_cache_stable_delta``: it only engages when the
prior original request is a message-prefix of the current one.

Real clients (litellm, Claude Code) move the ephemeral cache_control breakpoint
to the newest message every turn, so a historical message carries the marker on
one turn and not the next. The original raw-dict prefix compare therefore failed
every turn, dropping cache mode to RAW (uncompressed) forwarding -- byte-stable
(0 busts) but 0% compression. Observed directly on the mini-swe-agent cache-mode
run: avg_compression_pct=0.0 on every instance, orig==opt on every turn.

These tests pin the scenario against the real handler method and prove the
cache_control-agnostic compare lets the delta engage while the replayed prefix
stays byte-identical (so the provider prefix still hits).
"""
import copy

from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

delta = AnthropicHandlerMixin._extract_cache_stable_delta


def B(role, text, cc=False):
    """Anthropic block-style message; cache_control lives on a content block."""
    blk = {"type": "text", "text": text}
    if cc:
        blk["cache_control"] = {"type": "ephemeral"}
    return {"role": role, "content": [blk]}


# Turn t: client marked the (then-newest) msg2. We forwarded it verbatim.
PREV_ORIG = [B("user", "sys+task"), B("assistant", "ok"), B("user", "obs-1", cc=True)]
PREV_FWD = copy.deepcopy(PREV_ORIG)
# Turn t+1: appended act-2 + obs-2 and MOVED the marker off msg2 onto the newest.
CUR = [
    B("user", "sys+task"),
    B("assistant", "ok"),
    B("user", "obs-1"),  # marker gone
    B("assistant", "act-2"),
    B("user", "obs-2", cc=True),  # marker moved here
]


def test_moved_marker_engages_delta_not_raw_fallback():
    out = delta(CUR, PREV_ORIG, PREV_FWD)
    assert out is not None, "moved marker must NOT force raw fallback"
    stable_prefix, appended = out
    # The replayed prefix is byte-identical to what we forwarded (and the
    # provider cached) last turn -> the prefix hits instead of busting.
    assert stable_prefix == PREV_FWD
    # Only the two newly appended messages are handed to compression.
    assert len(appended) == 2
    assert appended[0]["content"][0]["text"] == "act-2"
    assert appended[1]["content"][0]["text"] == "obs-2"


def test_control_marker_not_moved_also_engages():
    # Same append, marker left on the historical msg2: engages either way.
    cur = [
        B("user", "sys+task"),
        B("assistant", "ok"),
        B("user", "obs-1", cc=True),
        B("assistant", "act-2"),
        B("user", "obs-2"),
    ]
    assert delta(cur, PREV_ORIG, PREV_FWD) is not None


def test_real_content_divergence_still_falls_back():
    # Safety preserved: a genuinely different historical message (not just a
    # moved marker) must still bail to raw -- we never replay stale content.
    cur = [
        B("user", "sys+task"),
        B("assistant", "DIFFERENT"),  # content actually changed
        B("user", "obs-1"),
        B("assistant", "act-2"),
    ]
    assert delta(cur, PREV_ORIG, PREV_FWD) is None


def test_cold_start_returns_none():
    assert delta(CUR, None, None) is None
    assert delta(CUR, [], []) is None


def test_shorter_current_returns_none():
    assert delta([B("user", "sys+task")], PREV_ORIG, PREV_FWD) is None
