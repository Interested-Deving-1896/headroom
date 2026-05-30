"""HeadroomEngine — request/response hook facade (Chunk 2 + 4.2a).

Composes the existing compression subsystems behind a clean hook interface.
Does NOT reimplement compression; delegates to injected ``CompressionPipeline``
instances via the ``ports.CompressionPipeline`` Protocol.

Design notes
------------
- **Dependency injection**: pipelines, config, usage_reporter are injected;
  no global state is read or written inside this module.
- **No silent fallbacks**: unregistered (provider, flavor) pairs raise loudly.
- **Passthrough fidelity**: when ``CompressionDecision.should_compress`` is
  False, ``on_request`` returns ``ctx.raw_body`` byte-identical (same object,
  no re-serialization).
- **Chunk 4.2a — real Anthropic path**: when ``anthropic_components`` is
  provided the engine orchestrates the full handler compression-core (mode
  branching, frozen-count, tool-sort, prepare_outbound_body_bytes) using the
  SAME callables the handler uses. CCR-tool-injection, memory injection, and
  proactive-expansion are excluded (Chunks 4.2b/4.2c).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from typing import Any

from headroom.engine.contract import (
    Flavor,
    Provider,
    RequestContext,
    RequestDecision,
    ResponseTelemetry,
    StreamContext,
)
from headroom.engine.ports import CompressionPipeline
from headroom.proxy.auth_mode import classify_auth_mode
from headroom.proxy.compression_decision import CompressionDecision
from headroom.transforms.compression_policy import resolve_policy


class AnthropicComponents:
    """Real Anthropic compression components for the engine.

    Replaces the fake-pipeline-only path when the engine should reproduce
    byte-identical output with the handler's compression-core path.

    Parameters
    ----------
    pipeline:
        The real ``TransformPipeline`` for Anthropic (same object the
        server builds in HeadroomProxy.__init__).
    provider:
        The AnthropicProvider (used for ``get_context_limit``).
    session_tracker_store:
        The ``SessionTrackerStore`` the engine owns (separate from the
        server's store so prefix-tracker state is engine-private).
    get_compression_cache:
        Callable ``(session_id: str) -> CompressionCache`` — same
        semantics as ``HeadroomProxy._get_compression_cache``.
    config:
        The ``ProxyConfig`` (mode, optimize, hooks, …).
    usage_reporter:
        Commercial gate for ``CompressionDecision.decide``.
    """

    def __init__(
        self,
        *,
        pipeline: Any,
        provider: Any,
        session_tracker_store: Any,
        get_compression_cache: Callable[[str], Any],
        config: Any,
        usage_reporter: Any | None,
    ) -> None:
        self.pipeline = pipeline
        self.provider = provider
        self.session_tracker_store = session_tracker_store
        self.get_compression_cache = get_compression_cache
        self.config = config
        self.usage_reporter = usage_reporter


class HeadroomEngine:
    """Facade that composes Headroom compression behind hook-shaped entry points.

    ``on_request`` is the load-bearing method. Two operating modes:

    **Fake-pipeline mode** (Chunks 1-2 tests, legacy): ``anthropic_components``
    is None; the engine uses ``pipelines`` to dispatch and applies a simplified
    (non-mode-branching) pipeline call. Existing Chunk 2 tests continue to pass
    because this path is unchanged.

    **Real-Anthropic mode** (Chunk 4.2a): ``anthropic_components`` is set.
    The engine owns the full compression-core orchestration for Anthropic
    requests: mode-branching (token/non-cache/cache-delta), frozen-count
    derivation, tool-sort, and ``prepare_outbound_body_bytes``. It faithfully
    reproduces what ``AnthropicHandlerMixin.handle_messages`` does for
    compression-core (excludes CCR injection / memory / proactive-expansion).

    Parameters
    ----------
    pipelines:
        Mapping from ``(Provider, Flavor)`` to a ``CompressionPipeline``
        implementor.  Fakes satisfy this in tests; used by the legacy path.
    config:
        Config object forwarded verbatim to ``CompressionDecision.decide``.
        Only ``config.optimize: bool`` is read there.
    usage_reporter:
        Commercial gate forwarded to ``CompressionDecision.decide``.
        ``None`` means no licensing → always allow compression.
    salt:
        Salt bytes for session key derivation (kept for CCR proactive-expansion
        wiring; not consumed in current chunks).
    anthropic_components:
        When set, the engine uses the real Anthropic orchestration path for
        Anthropic/Messages requests (Chunk 4.2a). When None, falls back to
        the fake-pipeline path (Chunks 1-2 behaviour).
    """

    def __init__(
        self,
        *,
        pipelines: Mapping[tuple[Provider, Flavor], CompressionPipeline],
        config: Any,
        usage_reporter: Any | None,
        salt: bytes,
        anthropic_components: AnthropicComponents | None = None,
    ) -> None:
        self._pipelines = dict(pipelines)
        self._config = config
        self._usage_reporter = usage_reporter
        self._salt = salt
        self._anthropic_components = anthropic_components

    # ── Request hook ──────────────────────────────────────────────────────────

    def on_request(self, ctx: RequestContext) -> RequestDecision:
        """Process an inbound request.

        For registered ``(provider, flavor)`` combos: classify auth mode,
        decide whether to compress, and either return the raw body unchanged
        (passthrough) or run the pipeline and return the mutated body.

        Raises
        ------
        KeyError
            If ``(ctx.provider, ctx.flavor)`` has no registered pipeline
            AND no real-component path handles it.
        ValueError
            If the raw body cannot be parsed as JSON (malformed request).
        """
        # Real Anthropic path (Chunk 4.2a)
        if (
            ctx.provider == Provider.ANTHROPIC
            and ctx.flavor == Flavor.MESSAGES
            and self._anthropic_components is not None
        ):
            return self._on_request_anthropic_real(ctx)

        # Legacy fake-pipeline path (Chunks 1-2)
        key = (ctx.provider, ctx.flavor)
        if key not in self._pipelines:
            raise KeyError(
                f"No pipeline registered for provider={ctx.provider!r}, "
                f"flavor={ctx.flavor!r}. Register it in the pipelines mapping."
            )

        return self._on_request_fake_pipeline(ctx, self._pipelines[key])

    # ── Real Anthropic orchestration (Chunk 4.2a) ─────────────────────────────

    def _on_request_anthropic_real(self, ctx: RequestContext) -> RequestDecision:
        """Reproduce the handler's compression-core path byte-for-byte.

        Mirrors ``AnthropicHandlerMixin.handle_messages`` compression-core:
        image compress → CompressionDecision → mode-branch pipeline.apply →
        tool-sort → prepare_outbound_body_bytes.

        Excluded (4.2b/4.2c): CCR tool injection, memory injection, proactive
        expansion, pipeline extension events, security scan, hooks.
        """
        from headroom.cache.compression_cache import CompressionCache  # noqa: F401
        from headroom.proxy.helpers import (
            BodyMutationTracker,
            prepare_outbound_body_bytes,
        )
        from headroom.proxy.image_compression_decision import ImageCompressionDecision
        from headroom.proxy.modes import is_cache_mode, is_token_mode
        from headroom.utils import extract_user_query

        ac = self._anthropic_components
        assert ac is not None

        original_body_bytes = ctx.raw_body

        # Parse body — raises loudly on malformed JSON
        try:
            body: dict[str, Any] = json.loads(original_body_bytes)
        except json.JSONDecodeError as exc:
            raise ValueError(f"on_request(anthropic): unparseable JSON body: {exc}") from exc

        messages: list[dict[str, Any]] = list(body.get("messages") or [])
        model: str = body.get("model", "unknown")
        # Preserve a deep copy of the original client messages (mirrors deep_copy
        # at handler line ~595) for use in the cache-delta path.
        original_client_messages: list[dict[str, Any]] = copy.deepcopy(messages)

        # Bypass: skip ALL compression when the caller explicitly opts out.
        headers = dict(ctx.headers_view)
        _bypass = (
            headers.get("x-headroom-bypass", "").lower() == "true"
            or headers.get("x-headroom-mode", "").lower() == "passthrough"
        )

        body_mutation_tracker = BodyMutationTracker()

        # Auth mode + policy (computed once; used by all three pipeline sites)
        auth_mode = classify_auth_mode(ctx.headers_view)
        compression_policy = resolve_policy(auth_mode)

        # Compression decision
        _decision = CompressionDecision.decide(
            headers=ctx.headers_view,
            config=ac.config,
            usage_reporter=ac.usage_reporter,
            messages=messages,
        )

        if not _decision.should_compress or _bypass:
            # Passthrough — return original bytes byte-identical.
            return RequestDecision(
                body=original_body_bytes,
                telemetry=ResponseTelemetry(compressed=False),
            )

        # --- Image compression (before text compression, same order as handler) ---
        _image_decision = ImageCompressionDecision.decide(
            headers=ctx.headers_view, config=ac.config, messages=messages
        )
        if _image_decision.should_compress and not is_cache_mode(ac.config.mode):
            from headroom.proxy.helpers import _get_image_compressor

            compressor = None
            try:
                compressor = _get_image_compressor()
                if compressor and compressor.has_images(messages):
                    messages = compressor.compress(messages, provider="anthropic")
                    body_mutation_tracker.mark_mutated("image_compression")
            finally:
                if compressor and hasattr(compressor, "close"):
                    compressor.close()

        # --- Session / frozen-count derivation ---
        # The engine owns its own session store (injected via AnthropicComponents);
        # the parity test seeds it with a controlled _FixedTracker just as the
        # golden recorder does.
        session_id = ac.session_tracker_store.compute_session_id(ctx, model, messages)
        prefix_tracker = ac.session_tracker_store.get_or_create(session_id, "anthropic")
        frozen_message_count = prefix_tracker.get_frozen_message_count()
        if is_cache_mode(ac.config.mode):
            # Mirrors _strict_previous_turn_frozen_count at handler line ~890.
            frozen_message_count = _strict_previous_turn_frozen_count(
                original_client_messages, frozen_message_count
            )

        # --- Context limit ---
        context_limit = ac.provider.get_context_limit(model)

        # --- hooks/biases (skipped in 4.2a — not present in golden corpus) ---
        biases = None
        request_id = ctx.request_id

        optimized_messages = messages

        # --- Mode branch: token / non-cache / cache-delta ---
        if is_token_mode(ac.config.mode):
            comp_cache = ac.get_compression_cache(session_id)

            # Zone 1: swap cached compressed versions into working copy
            working_messages = comp_cache.apply_cached(messages)

            # Clamp frozen_message_count (mirrors handler lines ~1039-1042)
            cache_frozen_count = comp_cache.compute_frozen_count(messages)
            frozen_message_count = min(frozen_message_count, cache_frozen_count)
            comp_cache.mark_stable_from_messages(messages, frozen_message_count)

            result = ac.pipeline.apply(
                messages=working_messages,
                model=model,
                model_limit=context_limit,
                context=extract_user_query(working_messages),
                frozen_message_count=frozen_message_count,
                biases=biases,
                request_id=request_id,
                compression_policy=compression_policy,
            )

            if result.messages != working_messages:
                comp_cache.update_from_result(messages, result.messages)

            optimized_messages = result.messages
            # Mirror handler line ~1064: always use pipeline result.
            # Structural diff check below detects any real mutation.

        elif not is_cache_mode(ac.config.mode):
            result = ac.pipeline.apply(
                messages=messages,
                model=model,
                model_limit=context_limit,
                context=extract_user_query(messages),
                frozen_message_count=frozen_message_count,
                biases=biases,
                request_id=request_id,
                compression_policy=compression_policy,
            )

            if result.messages != messages:
                optimized_messages = result.messages
                # Do NOT mark mutation explicitly here; structural diff below
                # detects the actual byte change. Handler mirrors this: no
                # explicit mark at lines ~1099-1104 for the non-cache path.

        else:
            # Cache-delta path
            previous_original_messages = prefix_tracker.get_last_original_messages()
            previous_forwarded_messages = prefix_tracker.get_last_forwarded_messages()
            delta = _extract_cache_stable_delta(
                original_client_messages,
                previous_original_messages,
                previous_forwarded_messages,
            )
            if delta is not None:
                stable_forwarded_prefix, delta_messages = delta
                if delta_messages:
                    result = ac.pipeline.apply(
                        messages=delta_messages,
                        model=model,
                        model_limit=context_limit,
                        context=extract_user_query(delta_messages),
                        frozen_message_count=0,
                        biases=biases,
                        request_id=request_id,
                        compression_policy=compression_policy,
                    )
                    optimized_messages = stable_forwarded_prefix + result.messages
                    # Mirror the handler: no explicit mark_mutated here.
                    # The structural diff check below will detect any real change.
                else:
                    optimized_messages = stable_forwarded_prefix
                    # No explicit mutation mark — structural diff detects if needed.
            else:
                # Conservative fallback for cache mode
                optimized_messages = messages

        # --- Tool sort (ALWAYS when tools present) ---
        tools = body.get("tools")
        if tools is not None:
            from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

            sorted_tools = AnthropicHandlerMixin._sort_tools_deterministically(tools)
            if sorted_tools != tools:
                body_mutation_tracker.mark_mutated("tool_sort")
            body["tools"] = sorted_tools

        # --- Reassemble body ---
        body["messages"] = optimized_messages

        # --- Structural mutation safety-net (mirrors handler lines ~1654-1660) ---
        if not body_mutation_tracker.mutated:
            try:
                parsed_original = json.loads(original_body_bytes)
                if parsed_original != body:
                    body_mutation_tracker.mark_mutated("structural_diff_vs_original")
            except (json.JSONDecodeError, ValueError):
                body_mutation_tracker.mark_mutated("original_unparseable")

        # --- Byte-faithful forward (mirrors prepare_outbound_body_bytes) ---
        outbound_bytes, _source = prepare_outbound_body_bytes(
            body=body,
            original_body_bytes=original_body_bytes,
            body_mutated=body_mutation_tracker.mutated,
        )

        compressed = body_mutation_tracker.mutated
        bytes_saved = max(0, len(original_body_bytes) - len(outbound_bytes))

        return RequestDecision(
            body=outbound_bytes,
            telemetry=ResponseTelemetry(
                bytes_saved=bytes_saved,
                compressed=compressed,
                ccr_fired=False,
            ),
        )

    # ── Legacy fake-pipeline path (Chunks 1-2) ────────────────────────────────

    def _on_request_fake_pipeline(
        self, ctx: RequestContext, pipeline: CompressionPipeline
    ) -> RequestDecision:
        """Simplified path used by Chunk 2 tests with fake pipelines.

        Preserves the original Chunk 2 semantics exactly so those tests
        continue passing.
        """
        # Parse body — raises loudly on malformed JSON
        try:
            body: dict[str, Any] = json.loads(ctx.raw_body)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"on_request: unparseable JSON body for "
                f"provider={ctx.provider!r}, flavor={ctx.flavor!r}: {exc}"
            ) from exc

        messages: list[dict[str, Any]] = body.get("messages") or []
        model: str = body.get("model", "")

        # Classify auth mode (pure, <10us, never raises)
        auth_mode = classify_auth_mode(ctx.headers_view)

        # Decision: should we compress?
        decision = CompressionDecision.decide(
            headers=ctx.headers_view,
            config=self._config,
            usage_reporter=self._usage_reporter,
            messages=messages,
        )

        if not decision.should_compress:
            # Return raw body BYTE-IDENTICAL — same object, no re-serialization.
            # This is load-bearing for prefix-cache safety.
            return RequestDecision(
                body=ctx.raw_body,
                telemetry=ResponseTelemetry(compressed=False),
            )

        # Resolve per-auth-mode compression policy
        policy = resolve_policy(auth_mode)

        # Delegate to the injected pipeline
        result = pipeline.apply(
            messages,
            model,
            compression_policy=policy,
        )

        # Reconstruct body with compressed messages
        body["messages"] = result.messages
        compressed_bytes = json.dumps(body).encode()

        bytes_saved = max(0, len(ctx.raw_body) - len(compressed_bytes))
        tokens_in = getattr(result, "tokens_before", 0)
        tokens_out = getattr(result, "tokens_after", 0)

        return RequestDecision(
            body=compressed_bytes,
            telemetry=ResponseTelemetry(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                bytes_saved=bytes_saved,
                compressed=True,
                ccr_fired=False,
            ),
        )

    # ── Response hooks (Chunk 2 stubs — Chunk 3+ extends these) ─────────────

    def on_response(self, ctx: RequestContext, raw_response: bytes) -> bytes:
        """Forward the upstream response unchanged.

        Chunk 3 will extend this with CCR proactive-expansion injection and
        token telemetry parsing.
        """
        return raw_response

    def on_response_chunk(self, sc: StreamContext, chunk: bytes) -> bytes:
        """Forward a streaming chunk unchanged.

        Chunk 3 will add SSE parsing for streaming token telemetry.
        """
        return chunk

    def on_response_end(self, sc: StreamContext, outcome: Any) -> ResponseTelemetry:
        """Finalize a streaming session and return its telemetry.

        Safe to call on normal completion OR abort (``outcome`` may be an
        Exception or ``None``).  Chunk 3 will accumulate streaming token
        counts here.
        """
        return ResponseTelemetry()


# ── Private helpers (mirrors static methods on AnthropicHandlerMixin) ─────────


def _strict_previous_turn_frozen_count(
    messages: list[dict[str, Any]],
    base_frozen_count: int,
) -> int:
    """Freeze all prior turns; only the final turn is mutable.

    Direct port of ``AnthropicHandlerMixin._strict_previous_turn_frozen_count``.
    """
    if not messages:
        return base_frozen_count
    final_idx = len(messages) - 1
    if messages[final_idx].get("role") == "user":
        return max(base_frozen_count, final_idx)
    return len(messages)


def _extract_cache_stable_delta(
    current_messages: list[dict[str, Any]],
    previous_original_messages: list[dict[str, Any]] | None,
    previous_forwarded_messages: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Return (stable_forwarded_prefix, appended_delta_messages) when safe.

    Direct port of ``AnthropicHandlerMixin._extract_cache_stable_delta``.
    """
    if not previous_original_messages or previous_forwarded_messages is None:
        return None
    prefix_len = len(previous_original_messages)
    if len(current_messages) < prefix_len:
        return None
    if current_messages[:prefix_len] != previous_original_messages:
        return None
    return (
        copy.deepcopy(previous_forwarded_messages),
        copy.deepcopy(current_messages[prefix_len:]),
    )
