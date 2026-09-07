"""Helpers that isolate litellm request shapes for compression.

The litellm proxy hands the guardrail a ``structured_messages`` list whose
content can be a plain string (OpenAI chat completions) or a list of typed
parts (Anthropic, OpenAI vision). lean-ctx only rewrites string content and
returns it as a plain string, so we:

* flatten list-of-parts content into a single string when every part is text;
* restore the original part shape afterwards, applying the rewritten string
  back to the first text part while preserving any non-text parts and any
  per-part fields like ``cache_control``.

Rows whose parts include images, audio or tool-call blocks are passed through
untouched: the compression service skips them, so the round-trip stays
identity-stable and the provider's prompt-cache prefix is not broken.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypeGuard

__all__ = [
    "BYPASS_HEADER",
    "LEAN_CTX_RETRIEVE_TOOL_NAME",
    "compressible_indices",
    "find_retrieve_tool_call_ids",
    "flatten_text_only_parts",
    "get_protected_indices",
    "is_all_text_parts",
    "merge_rewritten_text_parts",
    "parts_to_text",
    "restore_content_shapes",
    "retrieve_tool_call_name",
]


BYPASS_HEADER = "x-lean-ctx-bypass"
LEAN_CTX_RETRIEVE_TOOL_NAME = "lean_ctx_retrieve"


def is_all_text_parts(content: Any) -> TypeGuard[list[dict[str, Any]]]:
    """True only when ``content`` is a non-empty list of all-text parts.

    Strings short-circuit to False so the call site can use a single
    ``is_all_text_parts(content) or isinstance(content, str)`` check.
    """
    if not isinstance(content, list) or not content:
        return False
    return all(_is_text_part(part) for part in content)


def _is_text_part(part: Any) -> TypeGuard[dict[str, Any]]:
    if not isinstance(part, Mapping):
        return False
    # Both Anthropic ("type": "text") and OpenAI parts-without-type
    # shapes are text. Anything with a type that isn't "text" is skipped.
    part_type = part.get("type")
    if part_type is None:
        return isinstance(part.get("text"), str)
    if part_type == "text":
        return isinstance(part.get("text"), str)
    return False


def parts_to_text(parts: Iterable[Mapping[str, Any]]) -> str:
    """Join text-part ``text`` fields with no separator (Anthropic convention)."""
    chunks: list[str] = []
    for part in parts:
        text = part.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return "".join(chunks)


def merge_rewritten_text_parts(
    parts: list[Mapping[str, Any]],
    rewritten: str,
) -> list[dict[str, Any]]:
    """Apply ``rewritten`` back into the first text part of ``parts``.

    Non-text parts and any extra text parts are preserved verbatim so
    ``cache_control`` breakpoints, ordering, and any side-channel fields
    stay byte-identical. When ``rewritten`` matches the concatenation the
    parts already had, the original list is returned as-is so the call
    site can detect "this row was untouched".
    """
    out: list[dict[str, Any]] = []
    replaced = False
    for part in parts:
        if not replaced and _is_text_part(part):
            new_part = dict(part)
            new_part["text"] = rewritten
            out.append(new_part)
            replaced = True
        else:
            out.append(dict(part))
    if not replaced:
        # Defensive: caller should not invoke us unless ``parts`` has at
        # least one text part, but if they do, prepend the rewritten text
        # so the row's content is never silently empty.
        out.insert(0, {"type": "text", "text": rewritten})
    return out


def flatten_text_only_parts(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse all-text list-of-parts content into a plain string."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if is_all_text_parts(content):
            text = parts_to_text(content)
            if text:
                out.append({**msg, "content": text})
                continue
        out.append(msg)
    return out


def restore_content_shapes(
    originals: list[dict[str, Any]],
    returned: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Write compressed text back into each original row's content shape.

    Position-pairs original and returned rows. Returns ``returned`` as-is when
    the row count or any role diverges, because a reshaped conversation
    cannot be re-interleaved with the rows the compression service never saw.
    """
    if len(returned) != len(originals):
        return returned
    for orig, ret in zip(originals, returned, strict=False):
        if orig.get("role") != ret.get("role"):
            return returned
    restored: list[dict[str, Any]] = []
    for orig, ret in zip(originals, returned, strict=False):
        orig_content = orig.get("content")
        ret_content = ret.get("content")
        if isinstance(orig_content, list) and isinstance(ret_content, str):
            if ret_content == parts_to_text(
                [p for p in orig_content if isinstance(p, Mapping)]
            ):
                # Untouched row: keep the exact original parts, including
                # per-part fields like cache_control on later text parts.
                restored.append({**ret, "content": orig_content})
            else:
                restored.append(
                    {**ret, "content": merge_rewritten_text_parts(orig_content, ret_content)},
                )
        else:
            restored.append(ret)
    return restored


def _group_tool_exchanges(messages: SequenceLike) -> list[frozenset[int]]:
    """Group tool exchanges: assistant(tool_calls) rows plus their tool/function rows.

    Returns the exchanges as ordered frozensets of indices. The upstream headroom
    guardrail relies on the same grouping so a protected assistant tool call
    cannot end up answered by a marker standing in for the result the model
    just asked for.
    """
    groups: list[frozenset[int]] = []
    current: set[int] = set()
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            if current:
                groups.append(frozenset(current))
            current = {index}
        elif role in ("tool", "function"):
            current.add(index)
        else:
            if current:
                groups.append(frozenset(current))
                current = set()
    if current:
        groups.append(frozenset(current))
    return groups


def get_protected_indices(messages: SequenceLike) -> frozenset[int]:
    """Indices lean-ctx must not rewrite.

    Mirrors litellm's own compression policy: the system rows, the last user
    row, the last assistant row. The last user message is the instruction the
    model is being asked to act on, so rewriting it means the model answers a
    retrieval marker instead. The last assistant row anchors the agentic loop
    and is never touched.

    Any tool exchange that already overlaps a protected index is expanded
    wholesale (assistant(tool_calls) + its tool/function rows), matching the
    upstream headroom policy: a protected assistant tool call cannot end up
    answered by a marker standing in for the result the model just asked for.
    """
    protected: set[int] = set()
    last_user = -1
    last_assistant = -1
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "system":
            protected.add(index)
        elif role == "user":
            last_user = index
        elif role == "assistant":
            last_assistant = index
    if last_user >= 0:
        protected.add(last_user)
    if last_assistant >= 0:
        protected.add(last_assistant)
    expanded: set[int] = set(protected)
    for group in _group_tool_exchanges(messages):
        if group & protected:
            expanded |= group
    return frozenset(expanded)


def compressible_indices(
    messages: SequenceLike,
    protected: frozenset[int],
) -> list[int]:
    """Indices that should be sent to ``/v1/compress``.

    The returned list preserves the original ordering so positional
    interleaving with the protected rows stays straightforward.
    """
    return [i for i in range(len(messages)) if i not in protected]


def retrieve_tool_call_name(name: str | None) -> bool:
    """Match the retrieve tool whether called directly or via an MCP gateway.

    Server-side the tool is ``lean_ctx_retrieve``; exposed through LiteLLM's
    MCP gateway a client calls it as ``mcp__<server>__lean_ctx_retrieve``.
    """
    if name is None:
        return False
    return name == LEAN_CTX_RETRIEVE_TOOL_NAME or name.endswith(
        f"__{LEAN_CTX_RETRIEVE_TOOL_NAME}"
    )


def find_retrieve_tool_call_ids(messages: Iterable[Mapping[str, Any]]) -> frozenset[str]:
    """Tool-call ids of any ``lean_ctx_retrieve`` call in ``messages``.

    Reads both the OpenAI ``tool_calls`` shape and the Anthropic ``tool_use``
    parts so retrieval survives the provider translation that litellm runs.
    """
    ids: set[str] = set()
    for message in messages:
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, Mapping):
                    continue
                function = call.get("function")
                if not isinstance(function, Mapping):
                    continue
                if not retrieve_tool_call_name(function.get("name")):
                    continue
                call_id = call.get("id")
                if isinstance(call_id, str) and call_id:
                    ids.add(call_id)
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, Mapping):
                    continue
                if block.get("type") != "tool_use":
                    continue
                if not retrieve_tool_call_name(block.get("name")):
                    continue
                call_id = block.get("id")
                if isinstance(call_id, str) and call_id:
                    ids.add(call_id)
    return frozenset(ids)


def retrieval_result_indices(
    messages: SequenceLike,
    retrieve_call_ids: frozenset[str],
) -> frozenset[int]:
    """Indices of tool-result rows that carry ``lean_ctx_retrieve`` output.

    When the retrieve tool is exposed to a client that runs its own tool
    loop (the litellm MCP gateway path), the client executes the call and
    sends the recovered original content back as a tool result on the next
    turn. That content is exactly what a prior compression stubbed, so
    compressing it again re-derives the identical marker: a no-op that
    strands the model on the marker and loops the agent. Hold those rows
    back so the expansion survives.
    """
    if not retrieve_call_ids:
        return frozenset()
    out: set[int] = set()
    for index, message in enumerate(messages):
        role = message.get("role")
        if role not in ("tool", "function"):
            continue
        if str(message.get("tool_call_id")) in retrieve_call_ids:
            out.add(index)
    return frozenset(out)


# A ``SequenceLike`` accepts either a list of dicts or any object that
# supports ``__getitem__`` + ``__len__``. We use it as the signature for the
# pure helpers above so they stay trivially testable without a Sequence import.
SequenceLike = list[dict[str, Any]] | tuple[dict[str, Any], ...]
