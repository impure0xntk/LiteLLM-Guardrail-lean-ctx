"""Main ``LeanCTXGuardrail`` class and its litellm-proxy wiring.

The shape mirrors the upstream Headroom guardrail so a litellm-proxy admin
who already configured Headroom can swap ``headroom`` for ``lean-ctx`` by
changing the integration name and (optionally) the ``api_base``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.types.guardrails import GuardrailEventHooks, Mode

from .client import (
    DEFAULT_TIMEOUT_SECONDS,
    LeanCTXClient,
    LeanCTXHashMissing,
    LeanCTXHTTPError,
    LeanCTXUnreachable,
)
from .config import LeanCTXGuardrailConfigModel
from .messages import (
    BYPASS_HEADER,
    LEAN_CTX_RETRIEVE_TOOL_NAME,
    compressible_indices,
    find_retrieve_tool_call_ids,
    flatten_text_only_parts,
    get_protected_indices,
    restore_content_shapes,
    retrieval_result_indices,
    retrieve_tool_call_name,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.guardrails import Guardrail, LitellmParams
    from litellm.types.utils import (
        ChatCompletionToolParam,
        GenericGuardrailAPIInputs,
    )


__all__ = [
    "BYPASS_HEADER",
    "LEAN_CTX_RETRIEVE_TOOL_NAME",
    "LeanCTXGuardrail",
    "LeanCTXGuardrailConfigModel",
    "guardrail_class_registry",
    "guardrail_initializer_registry",
    "initialize_guardrail",
    "register",
]


logger = logging.getLogger(__name__)


GUARDRAIL_PROVIDER_NAME = "lean-ctx"
INTEGRATION_KEY = "lean-ctx"
HASH_CACHE_TTL_SECONDS = 15 * 60


@dataclass(slots=True)
class _CompressOutcome:
    """Internal bundle of a ``/v1/compress`` response plus timing."""

    succeeded: bool
    messages: list[dict[str, Any]]
    stats: dict[str, Any]
    ccr_hashes: frozenset[str]
    error: str | None = None
    detail: dict[str, Any] | None = None
    duration_seconds: float = 0.0


def _build_retrieve_tool() -> dict[str, Any]:
    """Return the tool spec we inject so the model can ask for an expansion."""
    return {
        "type": "function",
        "function": {
            "name": LEAN_CTX_RETRIEVE_TOOL_NAME,
            "description": (
                "Retrieve the original content that was compressed by lean-ctx. "
                "Call this when you encounter a compression marker containing a hash."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hash": {
                        "type": "string",
                        "description": "The hex hash from the compression marker.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional search query for BM25-ranked retrieval.",
                    },
                },
                "required": ["hash"],
            },
        },
    }


def _read_request_headers(request_data: Mapping[str, Any]) -> dict[str, str]:
    """Return the proxy_server_request headers as a flat dict, if any.

    The litellm proxy nests the original HTTP headers under
    ``proxy_server_request.headers``. We only need a few keys so we avoid
    importing the litellm-internal accessor and just walk the structure.
    """
    psr = request_data.get("proxy_server_request")
    if not isinstance(psr, Mapping):
        return {}
    headers = psr.get("headers")
    if not isinstance(headers, Mapping):
        return {}
    return {str(k): str(v) for k, v in headers.items()}


def _is_str_object_dict(value: object) -> bool:
    return isinstance(value, dict)


def _is_object_list(value: object) -> bool:
    return isinstance(value, list)


def _recalculate_full_context_stats(
    stats: dict[str, Any],
    *,
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    model: str | None,
) -> dict[str, Any]:
    """Recalculate compression statistics over the complete conversation."""
    try:
        import litellm

        tokens_before = litellm.token_counter(model=model, messages=before)
        tokens_after = litellm.token_counter(model=model, messages=after)
    except Exception:
        logger.debug("LeanCTX: unable to recalculate full-context token statistics", exc_info=True)
        return stats

    if not isinstance(tokens_before, int) or not isinstance(tokens_after, int):
        return stats
    if tokens_before <= 0:
        return stats

    return {
        **stats,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "tokens_saved": tokens_before - tokens_after,
        "compression_ratio": tokens_after / tokens_before,
    }


def _retrieve_call_ids_from_request(
    request_data: Mapping[str, Any],
) -> frozenset[str]:
    """Find ``lean_ctx_retrieve`` tool-call ids in the request's own messages.

    The translation handler truncates a tool name over 64 chars to
    ``{prefix}_{hash}``, which drops the ``__lean_ctx_retrieve`` suffix a long
    ``mcp__<server>__`` prefix pushes past the limit. Tool-call ids are
    never truncated, so reading the untranslated messages here keeps the
    pairing intact.
    """
    raw_messages: object = request_data.get("messages")
    if not isinstance(raw_messages, list):
        return frozenset()
    # ``find_retrieve_tool_call_ids`` walks Mapping-shaped rows; the litellm
    # adapter's own ``AllMessageValues`` TypedDicts qualify structurally so a
    # narrowing cast keeps the type checker honest.
    dict_rows = [row for row in raw_messages if isinstance(row, dict)]
    return find_retrieve_tool_call_ids(dict_rows)


def _coerce_event_hook(
    mode: str | list[str] | Mode,
) -> GuardrailEventHooks | list[GuardrailEventHooks] | Mode:
    if isinstance(mode, Mode):
        return mode
    if isinstance(mode, list):
        return [GuardrailEventHooks(item) for item in mode]
    return GuardrailEventHooks(mode)


def _resolve_call_id(
    logging_obj: object | None,
    request_state: Mapping[str, Any],
) -> str | None:
    """Pick the ``litellm_call_id`` shared by a request's pre-call hook and
    its agentic-loop hooks, so CCR hash validation is scoped per call."""
    logging_call_id = getattr(logging_obj, "litellm_call_id", None)
    if isinstance(logging_call_id, str) and logging_call_id:
        return logging_call_id
    kwargs_call_id = request_state.get("litellm_call_id")
    return kwargs_call_id if isinstance(kwargs_call_id, str) else None



def _extract_retrieve_tool_calls(response: object) -> list[dict[str, Any]]:
    """Pull out ``lean_ctx_retrieve`` tool calls from a model response.

    Walks the three shapes the litellm adapter hands back: a top-level
    ``tool_calls`` field (OpenAI Responses API), ``choices[0].message.tool_calls``
    (OpenAI Chat Completions), and ``content`` blocks with ``type == tool_use``
    (Anthropic Messages API). Anything that does not name the retrieve tool is
    skipped so unrelated tool calls in the same turn do not short-circuit.
    """
    out: list[dict[str, Any]] = []

    candidates: list[object] = []
    top_level = getattr(response, "tool_calls", None) or _maybe_get(response, "tool_calls")
    if isinstance(top_level, list):
        candidates.extend(top_level)

    content_blocks = _maybe_get(response, "content")
    if isinstance(content_blocks, list):
        for block in content_blocks:
            if isinstance(block, Mapping) and block.get("type") == "tool_use":
                candidates.append(block)

    choices = _maybe_get(response, "choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        message = _maybe_get(first, "message") if isinstance(first, Mapping) else getattr(first, "message", None)
        if message is not None:
            chat_calls = _maybe_get(message, "tool_calls")
            if isinstance(chat_calls, list):
                candidates.extend(chat_calls)

    for call in candidates:
        # Accept both Mapping (dict) and object-shaped tool calls; the litellm
        # adapter sometimes hands back SimpleNamespace, litellm ModelResponse,
        # or a pydantic BaseModel and the consumer should not care which.
        if isinstance(call, Mapping):
            def get(key, _c=call):
                return _c.get(key)
        elif hasattr(call, "__getitem__"):
            subscript_view = cast("Mapping[Any, Any]", call)

            def get(key, _v=subscript_view):
                try:
                    return _v[key]
                except Exception:
                    return None
        else:
            def get(key, _c=call):
                return getattr(_c, key, None)
        function = get("function")
        if isinstance(function, Mapping):
            def get_fn(key, _f=function, default=None):
                return _f.get(key, default)
        else:
            def get_fn(key, _f=function, default=None):
                return getattr(_f, key, default)
        name = get_fn("name")
        arguments = get_fn("arguments", "{}")
        call_id = get("id")
        if not retrieve_tool_call_name(name if isinstance(name, str) else None):
            continue
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        try:
            parsed_args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            parsed_args = {}
        if not isinstance(parsed_args, dict):
            parsed_args = {}
        out.append(
            {
                "id": call_id,
                "name": name,
                "arguments": parsed_args,
            }
        )
    return out


def _maybe_get(obj: object, key: str) -> object | None:
    """Dict-or-object accessor used for response objects that may be either."""
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _response_has_output_list(response: object) -> bool:
    return isinstance(_maybe_get(response, "output"), list)


def _response_has_anthropic_content_list(response: object) -> bool:
    return isinstance(_maybe_get(response, "content"), list)


def _assistant_text_from_response(response: object) -> str:
    """Extract any assistant text from a model response, across providers."""
    # OpenAI Responses API
    output = _maybe_get(response, "output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") in {"message", "text"}:
                content = item.get("content")
                if isinstance(content, str):
                    chunks.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                            chunks.append(part["text"])
        if chunks:
            return "".join(chunks)
    # Anthropic Messages API
    content = _maybe_get(response, "content")
    if isinstance(content, list):
        chunks = []
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        if chunks:
            return "".join(chunks)
    # OpenAI Chat Completions
    choices = _maybe_get(response, "choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        message = _maybe_get(first, "message") if isinstance(first, Mapping) else getattr(first, "message", None)
        if message is not None:
            content = _maybe_get(message, "content")
            if isinstance(content, str):
                return content
    return ""


class LeanCTXGuardrail(CustomGuardrail):
    """LiteLLM custom guardrail that compresses via the lean-ctx proxy.

    Wire it into litellm-proxy by setting the integration name to
    ``lean-ctx`` and pointing ``api_base`` at the lean-ctx server
    (default ``http://localhost:4444``).
    """

    records_own_guardrail_information: ClassVar[bool] = True

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        guardrail_name: str | None = None,
        event_hook: GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None = None,
        default_on: bool = False,
        unreachable_fallback: Literal["fail_closed", "fail_open"] | None = None,
        timeout: float | None = None,
        ccr_retrieval: bool = True,
        logging: bool = False,
        **kwargs: Any,
    ) -> None:
        import os

        # ``litellm.proxy.guardrails.guardrail_registry.initialize_custom_guardrail``
        # forwards every field from ``LitellmParams.model_dump(exclude_none=True)``
        # via ``**extra_params``. ``LitellmParams`` mixes in models for the other
        # built-in guardrails (Bedrock, Presidio, Lakera, Javelin, ...) so the
        # dump includes fields this class does not understand (``version``,
        # ``presidio_language``, ``block_on_violation``, ...). Drop the unknown
        # ones instead of letting ``TypeError`` abort proxy startup, but log
        # them at debug so a misconfiguration is still observable.
        unknown_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"guardrail", "mode"}
        }
        if unknown_kwargs:
            logger.debug(
                "LeanCTXGuardrail ignoring unknown kwargs: %s",
                sorted(unknown_kwargs),
            )

        resolved_api_base = (
            api_base
            or os.environ.get("LEAN_CTX_API_BASE")
            or os.environ.get("LEANCTX_API_BASE")
        )
        if not resolved_api_base:
            raise ValueError(
                "LeanCTX guardrail requires an API base URL. "
                "Set `api_base` in the guardrail config, or the LEAN_CTX_API_BASE env var."
            )
        resolved_api_key = (
            api_key
            or os.environ.get("LEAN_CTX_API_KEY")
            or os.environ.get("LEANCTX_API_KEY")
        )
        self._client = LeanCTXClient(
            resolved_api_base,
            api_key=resolved_api_key,
            timeout=timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS,
        )
        self.lean_ctx_model = model
        self.unreachable_fallback: Literal["fail_closed", "fail_open"] = (
            "fail_open" if unreachable_fallback == "fail_open" else "fail_closed"
        )
        self.ccr_retrieval = ccr_retrieval
        self.logging_enabled = logging
        # hash registry: litellm_call_id -> (frozenset[hash], expiry monotonic).
        # A forged hash-shaped string from another request must not pass CCR
        # validation, so we scope every hash by the call id that produced it.
        self._issued_hashes_by_call_id: dict[str, tuple[frozenset[str], float]] = {}
        super().__init__(
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
            supported_event_hooks=list(self.get_supported_event_hooks()),
        )

    # ----- internal helpers -------------------------------------------------

    def _should_bypass(self, request_data: Mapping[str, Any]) -> bool:
        headers = _read_request_headers(request_data)
        value = headers.get(BYPASS_HEADER)
        return value is not None and value.lower() == "true"

    def _prune_expired_hashes(self) -> None:
        now = time.monotonic()
        self._issued_hashes_by_call_id = {
            call_id: (hashes, expiry)
            for call_id, (hashes, expiry) in self._issued_hashes_by_call_id.items()
            if expiry > now
        }

    async def _call_compress(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None,
    ) -> _CompressOutcome:
        start = time.time()
        try:
            result = await self._client.compress(messages, model=model)
        except LeanCTXUnreachable as exc:
            return _CompressOutcome(
                succeeded=False,
                messages=list(messages),
                stats={},
                ccr_hashes=frozenset(),
                error="lean-ctx compression service unreachable",
                detail={"detail": str(exc)},
                duration_seconds=time.time() - start,
            )
        except LeanCTXHTTPError as exc:
            return _CompressOutcome(
                succeeded=False,
                messages=list(messages),
                stats={},
                ccr_hashes=frozenset(),
                error="lean-ctx compression service returned an error",
                detail=self._build_failure_detail(exc),
                duration_seconds=time.time() - start,
            )
        return _CompressOutcome(
            succeeded=True,
            messages=result.messages,
            stats=result.stats.as_dict(),
            ccr_hashes=result.ccr_hashes,
            duration_seconds=time.time() - start,
        )

    @staticmethod
    def _build_failure_detail(exc: LeanCTXHTTPError) -> dict[str, Any]:
        detail: dict[str, Any] = {"status_code": exc.status_code, "body": exc.body}
        if exc.status_code == 404:
            detail["hint"] = (
                "The lean-ctx compression endpoint returned HTTP 404. "
                "Verify that the configured lean-ctx endpoint is correct and that "
                "the compression endpoint is reachable. Lean-ctx serves /v1/compress "
                "on the port printed by `lean-ctx proxy status` (default 4444)."
            )
        return detail

    def _handle_compress_failure(
        self,
        messages: list[dict[str, Any]],
        outcome: _CompressOutcome,
    ) -> list[dict[str, Any]]:
        if self.unreachable_fallback == "fail_open":
            logger.warning(
                "LeanCTX: %s; fail_open configured, forwarding request uncompressed. detail=%s",
                outcome.error,
                outcome.detail,
            )
            return list(messages)
        raise LeanCTXGatewayError(outcome.error or "lean-ctx compression failed", outcome.detail or {})

    # ----- litellm hooks ----------------------------------------------------

    @log_guardrail_information
    async def apply_guardrail(  # type: ignore[override]
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        if input_type != "request":
            return inputs

        if self._should_bypass(request_data):
            logger.debug("LeanCTX: %s header set; skipping compression", BYPASS_HEADER)
            return inputs

        if request_data.get("background"):
            logger.debug("LeanCTX: background request; skipping compression")
            return inputs

        structured_messages = inputs.get("structured_messages")
        if not _is_object_list(structured_messages) or not structured_messages:
            return inputs

        # ``structured_messages`` is a list of litellm TypedDicts (e.g.
        # ``ChatCompletionUserMessage``); TypedDicts are not ``dict`` subtypes
        # for ty, but they are structurally dict-shaped for ``_is_str_object_dict``
        # so we narrow explicitly with a cast to keep the rest of the pipeline
        # type-safe.
        raw_rows: list[dict[str, Any]] = cast(
            "list[dict[str, Any]]",
            [row for row in (structured_messages or []) if _is_str_object_dict(row)],
        )
        if not raw_rows:
            return inputs
        messages = raw_rows

        # The last user message is the instruction the model is being asked to
        # act on, so replacing it with a marker means the model answers a
        # retrieval result instead of the request. Protected rows are held
        # back from the payload rather than pinned after the fact, so their
        # tokens are not counted as savings we never apply.
        raw_retrieve_ids = _retrieve_call_ids_from_request(request_data)
        protected = get_protected_indices(messages) | retrieval_result_indices(
            messages, raw_retrieve_ids
        )
        compressible_rows = compressible_indices(messages, protected)
        if not compressible_rows:
            return inputs
        compressible = [messages[i] for i in compressible_rows]

        model: Any = self.lean_ctx_model or request_data.get("model")
        model_name = model if isinstance(model, str) else None

        outcome = await self._call_compress(
            flatten_text_only_parts(compressible),
            model=model_name,
        )

        if not outcome.succeeded:
            # Spend tracking still benefits from a failed-call record; the
            # standard logging machinery expects either a stats dict or None,
            # so we pass an explicit error payload.
            self._record_failure(request_data, outcome)
            if self.unreachable_fallback == "fail_open":
                return inputs
            raise LeanCTXGatewayError(
                outcome.error or "lean-ctx compression failed",
                outcome.detail or {},
            )

        compressed_compressible = restore_content_shapes(
            compressible,
            outcome.messages,
        )
        sent_positions = [i for i in range(len(messages)) if i not in protected]
        compressed_by_index = dict(zip(sent_positions, compressed_compressible, strict=False))
        rewritten = [
            messages[i] if i in protected else compressed_by_index[i]
            for i in range(len(messages))
        ]
        outcome.stats = _recalculate_full_context_stats(
            outcome.stats,
            before=messages,
            after=rewritten,
            model=model_name,
        )

        self._record_success(request_data, outcome)

        if not self.ccr_retrieval or not outcome.ccr_hashes:
            # ``inputs`` is a TypedDict (``GenericGuardrailAPIInputs``); we hand
            # back a fresh dict with the rewritten rows cast to the same shape
            # so the litellm consumer does not see a wider type.
            return cast(
                "GenericGuardrailAPIInputs",
                {**inputs, "structured_messages": rewritten},
            )

        # Record the hashes issued for this call so the agentic-loop hook can
        # reject forged hash-shaped strings planted in a user prompt.
        self._prune_expired_hashes()
        call_id = _resolve_call_id(logging_obj, request_data)
        if not call_id:
            call_id = str(uuid.uuid4())
            request_data["litellm_call_id"] = call_id
        self._issued_hashes_by_call_id[call_id] = (
            frozenset(outcome.ccr_hashes),
            time.monotonic() + HASH_CACHE_TTL_SECONDS,
        )

        existing_tools = inputs.get("tools")
        retrieve_tool = _build_retrieve_tool()
        # The ``ChatCompletionToolParam`` TypedDict is structurally compatible
        # with our dict-built tool spec; the litellm consumer accepts both.
        retrieve_tool_param = cast("ChatCompletionToolParam", retrieve_tool)
        if isinstance(existing_tools, list):
            merged_tools = list(existing_tools)
            if not any(_is_retrieve_tool(t) for t in merged_tools):
                merged_tools.append(retrieve_tool_param)
        else:
            merged_tools = [retrieve_tool_param]

        return cast(
            "GenericGuardrailAPIInputs",
            {
                **inputs,
                "structured_messages": rewritten,
                "tools": merged_tools,
            },
        )

    # ----- standard logging integration -------------------------------------

    def _emit_compression_log(
        self,
        *,
        outcome: _CompressOutcome,
        status: str,
    ) -> None:
        if not self.logging_enabled:
            return
        logger.info(
            "LeanCTX guardrail compression status=%s duration_seconds=%.3f "
            "tokens_before=%s tokens_after=%s tokens_saved=%s ccr_hashes=%d error=%s",
            status,
            outcome.duration_seconds,
            outcome.stats.get("tokens_before"),
            outcome.stats.get("tokens_after"),
            outcome.stats.get("tokens_saved"),
            len(outcome.ccr_hashes),
            outcome.error,
        )

    def _record_success(
        self,
        request_data: dict,
        outcome: _CompressOutcome,
    ) -> None:
        try:
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response=outcome.stats,
                request_data=request_data,
                guardrail_status="success",
                guardrail_provider=GUARDRAIL_PROVIDER_NAME,
                event_type=GuardrailEventHooks.pre_call,
                start_time=time.time() - outcome.duration_seconds,
                end_time=time.time(),
                duration=outcome.duration_seconds,
            )
        except Exception:  # pragma: no cover - best-effort logging
            logger.debug("LeanCTX: failed to record success log", exc_info=True)
        self._emit_compression_log(outcome=outcome, status="success")
        try:
            from litellm.proxy.common_utils.callback_utils import (
                add_guardrail_to_applied_guardrails_header,
            )
            add_guardrail_to_applied_guardrails_header(
                request_data=request_data,
                guardrail_name=self.guardrail_name,
            )
        except Exception:  # pragma: no cover - litellm-internal optional
            pass

    def _record_failure(
        self,
        request_data: dict,
        outcome: _CompressOutcome,
    ) -> None:
        try:
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response={
                    "error": outcome.error,
                    **(outcome.detail or {}),
                },
                request_data=request_data,
                guardrail_status="guardrail_failed_to_respond",
                guardrail_provider=GUARDRAIL_PROVIDER_NAME,
                event_type=GuardrailEventHooks.pre_call,
                start_time=time.time() - outcome.duration_seconds,
                end_time=time.time(),
                duration=outcome.duration_seconds,
            )
        except Exception:  # pragma: no cover - best-effort logging
            logger.debug("LeanCTX: failed to record failure log", exc_info=True)
        self._emit_compression_log(outcome=outcome, status="failure")
        try:
            from litellm.proxy.common_utils.callback_utils import (
                add_guardrail_to_applied_guardrails_header,
            )
            add_guardrail_to_applied_guardrails_header(
                request_data=request_data,
                guardrail_name=self.guardrail_name,
            )
        except Exception:  # pragma: no cover - litellm-internal optional
            pass

    # ----- agentic loop (CCR retrieval) -------------------------------------

    async def async_should_run_agentic_loop(  # type: ignore[override]
        self,
        response: Any,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        stream: bool,
        custom_llm_provider: str,
        kwargs: dict,
    ) -> tuple[bool, dict]:
        if not tools or not any(_is_retrieve_tool(t) for t in tools):
            return False, {}
        tool_calls = _extract_retrieve_tool_calls(response)
        if not tool_calls:
            return False, {}
        return True, {"tool_calls": tool_calls}

    async def async_build_agentic_loop_plan(  # type: ignore[override]
        self,
        tools: dict,
        model: str,
        messages: list[dict],
        response: Any,
        anthropic_messages_provider_config: Any,
        anthropic_messages_optional_request_params: dict,
        logging_obj: LiteLLMLoggingObj | None,
        stream: bool,
        kwargs: dict,
    ) -> Any:
        # The AgenticLoopPlan TypedDict is optional at runtime; we import it
        # only when actually building a plan so unit tests can run without it.
        from litellm.types.integrations.custom_logger import (
            AgenticLoopPlan,
            AgenticLoopRequestPatch,
        )

        tool_calls: list[dict[str, Any]] = list(tools.get("tool_calls", []))

        self._prune_expired_hashes()
        call_id = _resolve_call_id(logging_obj, kwargs)
        valid_hashes = (
            self._issued_hashes_by_call_id.get(call_id, (frozenset(), 0.0))[0]
            if call_id
            else frozenset()
        )

        retrieved: list[tuple[dict[str, Any], str]] = []
        for tc in tool_calls:
            raw_arguments = tc.get("arguments")
            arguments: dict[str, Any] = raw_arguments if isinstance(raw_arguments, Mapping) else {}
            raw_hash = arguments.get("hash", "")
            hash_value = str(raw_hash).lower()
            query = arguments.get("query")
            if hash_value not in valid_hashes:
                logger.warning(
                    "LeanCTX CCR: rejecting hash=%s not produced by current request compression",
                    hash_value,
                )
                content = f"[LeanCTX: hash={hash_value} was not produced by the current request]"
            else:
                content = await self._call_retrieve(
                    hash_value=hash_value,
                    query=str(query) if isinstance(query, str) and query else None,
                )
            retrieved.append((tc, content))

        if _response_has_output_list(response):
            follow_up = list(messages) + _build_responses_followup_items(response, retrieved)
        elif _response_has_anthropic_content_list(response):
            follow_up = list(messages) + _build_anthropic_followup_messages(response, retrieved)
        else:
            assistant_message = _build_assistant_message_from_response(response, retrieved)
            tool_results = [
                {"role": "tool", "tool_call_id": tc.get("id"), "content": content}
                for tc, content in retrieved
            ]
            follow_up = [*messages, assistant_message, *tool_results]

        optional_params = {
            k: v
            for k, v in anthropic_messages_optional_request_params.items()
            if k != "max_tokens"
        }
        max_tokens = anthropic_messages_optional_request_params.get("max_tokens") or kwargs.get("max_tokens")

        full_model_name = model
        if logging_obj is not None:
            agentic_params = getattr(logging_obj, "model_call_details", {}) or {}
            agentic_params = agentic_params.get("agentic_loop_params", {}) or {}
            candidate = agentic_params.get("model", model)
            if isinstance(candidate, str) and candidate:
                full_model_name = candidate

        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=AgenticLoopRequestPatch(
                model=full_model_name,
                messages=follow_up,
                max_tokens=max_tokens,
                optional_params=optional_params,
                kwargs={
                    k: v
                    for k, v in kwargs.items()
                    if not k.startswith("_lean_ctx") and k != "litellm_logging_obj"
                },
            ),
            metadata={"tool_type": "lean_ctx_ccr"},
        )

    async def _call_retrieve(self, *, hash_value: str, query: str | None) -> str:
        try:
            return await self._client.retrieve(hash_value, query=query)
        except LeanCTXHashMissing:
            return f"[LeanCTX: hash={hash_value} not found or expired]"
        except (LeanCTXHTTPError, LeanCTXUnreachable) as exc:
            logger.warning("LeanCTX: retrieve failed for hash=%s: %s", hash_value, exc)
            return f"[LeanCTX: retrieval failed for hash={hash_value}]"

    # ----- config exposure --------------------------------------------------

    @staticmethod
    def get_config_model() -> type[LeanCTXGuardrailConfigModel] | None:
        return LeanCTXGuardrailConfigModel


def _is_retrieve_tool(tool: Any) -> bool:
    if not isinstance(tool, Mapping):
        return False
    function = tool.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        return name == LEAN_CTX_RETRIEVE_TOOL_NAME
    return tool.get("name") == LEAN_CTX_RETRIEVE_TOOL_NAME


def _build_assistant_message_from_response(
    response: object,
    retrieved: list[tuple[dict[str, Any], str]],
) -> dict[str, Any]:
    text = _assistant_text_from_response(response)
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {
                "id": tc.get("id"),
                "type": "function",
                "function": {
                    "name": tc.get("name"),
                    "arguments": json.dumps(tc.get("arguments", {})),
                },
            }
            for tc, _ in retrieved
        ],
    }


def _build_anthropic_followup_messages(
    response: object,
    retrieved: list[tuple[dict[str, Any], str]],
) -> list[dict[str, Any]]:
    """Build Anthropic Messages API follow-up messages for a tool round-trip."""
    text = _assistant_text_from_response(response)
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": ([{"type": "text", "text": text}] if text else [])
        + [
            {
                "type": "tool_use",
                "id": tc.get("id"),
                "name": tc.get("name"),
                "input": tc.get("arguments", {}),
            }
            for tc, _ in retrieved
        ],
    }
    user_message: dict[str, Any] = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tc.get("id"), "content": content}
            for tc, content in retrieved
        ],
    }
    return [assistant_message, user_message]


def _build_responses_followup_items(
    response: object,
    retrieved: list[tuple[dict[str, Any], str]],
) -> list[dict[str, Any]]:
    """Build OpenAI Responses API follow-up input items for a tool round-trip."""
    text = _assistant_text_from_response(response)
    items: list[dict[str, Any]] = [{"role": "assistant", "content": text}] if text else []
    for tc, content in retrieved:
        call_id = tc.get("id")
        items.append(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": tc.get("name"),
                "arguments": json.dumps(tc.get("arguments", {})),
            }
        )
        items.append({"type": "function_call_output", "call_id": call_id, "output": content})
    return items


# ----- litellm-proxy integration ------------------------------------------


class LeanCTXGatewayError(RuntimeError):
    """Raised when the lean-ctx compression service fails and fallback is closed."""

    def __init__(self, message: str, detail: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.detail = dict(detail)


def initialize_guardrail(
    litellm_params: LitellmParams,
    guardrail: Guardrail,
) -> LeanCTXGuardrail:
    """litellm-proxy callback: build and register a ``LeanCTXGuardrail``."""
    import litellm

    callback = LeanCTXGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        model=litellm_params.model,
        guardrail_name=guardrail["guardrail_name"],
        event_hook=_coerce_event_hook(litellm_params.mode),
        default_on=litellm_params.default_on or False,
        unreachable_fallback=litellm_params.unreachable_fallback,
        timeout=litellm_params.timeout,
        ccr_retrieval=getattr(litellm_params, "ccr_retrieval", True),
        logging=getattr(litellm_params, "logging", False),
    )
    litellm.logging_callback_manager.add_litellm_callback(callback)
    return callback


# ``litellm.proxy.guardrails.guardrail_hooks`` looks these up by string key
# when the proxy loads ``litellm_params.guardrail == "lean-ctx"``. We expose
# the registries as module-level globals so they can be imported into the
# ``litellm.guardrail_initializer_registry`` / ``guardrail_class_registry``
# from a single entry point.

guardrail_class_registry: dict[str, type[LeanCTXGuardrail]] = {
    INTEGRATION_KEY: LeanCTXGuardrail,
}

guardrail_initializer_registry: dict[str, Any] = {
    INTEGRATION_KEY: initialize_guardrail,
}


def register() -> None:
    """Register this guardrail with litellm's global registries.

    litellm exposes mutable module-level dicts that the proxy reads at
    startup. Calling this once at process boot wires up the integration
    under the name ``lean-ctx``.
    """
    import litellm  # local import so test suites can import the package first

    litellm.guardrail_class_registry.update(guardrail_class_registry)
    litellm.guardrail_initializer_registry.update(guardrail_initializer_registry)
