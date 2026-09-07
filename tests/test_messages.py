"""Tests for the pure message-shape helpers."""

from __future__ import annotations

from litellm_guardrail_lean_ctx.messages import (
    compressible_indices,
    find_retrieve_tool_call_ids,
    flatten_text_only_parts,
    get_protected_indices,
    is_all_text_parts,
    merge_rewritten_text_parts,
    parts_to_text,
    restore_content_shapes,
    retrieval_result_indices,
    retrieve_tool_call_name,
)


def test_is_all_text_parts_handles_both_shapes() -> None:
    assert is_all_text_parts("plain string") is False
    assert is_all_text_parts([]) is False
    assert is_all_text_parts(
        [{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}]
    )
    assert not is_all_text_parts(
        [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {}}]
    )


def test_parts_to_text_concatenates_in_order() -> None:
    text = parts_to_text(
        [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
            {"type": "text", "text": "c"},
        ]
    )
    assert text == "abc"


def test_merge_rewritten_text_parts_preserves_non_text_and_cache_control() -> None:
    parts = [
        {"type": "text", "text": "head ", "cache_control": {"type": "ephemeral"}},
        {"type": "image_url", "image_url": {"url": "localhost"}},
        {"type": "text", "text": " tail"},
    ]
    rewritten = merge_rewritten_text_parts(parts, "head tail")
    assert rewritten[0]["text"] == "head tail"
    assert rewritten[0]["cache_control"] == {"type": "ephemeral"}
    assert rewritten[1] == parts[1]
    assert rewritten[2] == parts[2]


def test_flatten_text_only_parts_passes_images_through() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "x"}},
            ],
        }
    ]
    out = flatten_text_only_parts(messages)
    assert out[0]["content"] == messages[0]["content"]


def test_flatten_text_only_parts_collapses_all_text() -> None:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
    ]
    out = flatten_text_only_parts(messages)
    assert out[0]["content"] == "ab"


def test_restore_content_shapes_keeps_untouched_rows_intact() -> None:
    originals = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "b"},
            ],
        }
    ]
    returned = [{"role": "user", "content": "ab"}]
    restored = restore_content_shapes(originals, returned)
    assert restored[0]["content"] == originals[0]["content"]


def test_restore_content_shapes_applies_rewritten_text() -> None:
    originals = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "head"},
                {"type": "text", "text": "tail"},
            ],
        }
    ]
    returned = [{"role": "user", "content": "headTAIL"}]
    restored = restore_content_shapes(originals, returned)
    assert restored[0]["content"][0]["text"] == "headTAIL"
    assert restored[0]["content"][1] == originals[0]["content"][1]


def test_restore_content_shapes_adopts_returned_on_drift() -> None:
    originals = [
        {"role": "system", "content": "x"},
        {"role": "user", "content": "y"},
    ]
    returned = [{"role": "user", "content": "only-user"}]
    # Row count differs; we cannot re-interleave, so we trust the service.
    restored = restore_content_shapes(originals, returned)
    assert restored == returned


def test_get_protected_indices_anchors_system_and_last_user_assistant() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "third"},
    ]
    protected = get_protected_indices(messages)
    assert protected == frozenset({0, 5, 4})


def test_compressible_indices_excludes_protected() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ]
    protected = frozenset({0, 2})
    assert compressible_indices(messages, protected) == [1]


def test_retrieve_tool_call_name_matches_bare_and_mcp_prefix() -> None:
    assert retrieve_tool_call_name("lean_ctx_retrieve")
    assert retrieve_tool_call_name("mcp__lean__lean_ctx_retrieve")
    assert not retrieve_tool_call_name("other_tool")
    assert retrieve_tool_call_name(None) is False


def test_find_retrieve_tool_call_ids_reads_openai_and_anthropic_shapes() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "openai-1",
                    "function": {"name": "lean_ctx_retrieve"},
                },
                {
                    "id": "openai-2",
                    "function": {"name": "lean_ctx_retrieve", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "thinking"},
                {"type": "tool_use", "id": "anthropic-1", "name": "lean_ctx_retrieve", "input": {}},
            ],
        },
    ]
    ids = find_retrieve_tool_call_ids(messages)
    assert ids == frozenset({"openai-1", "openai-2", "anthropic-1"})


def test_retrieval_result_indices_matches_tool_call_id() -> None:
    messages = [
        {"role": "tool", "tool_call_id": "openai-1", "content": "result"},
        {"role": "tool", "tool_call_id": "other", "content": "result"},
    ]
    indices = retrieval_result_indices(messages, frozenset({"openai-1"}))
    assert indices == frozenset({0})


def test_retrieval_result_indices_empty_when_no_calls() -> None:
    messages = [{"role": "tool", "tool_call_id": "x", "content": "y"}]
    assert retrieval_result_indices(messages, frozenset()) == frozenset()
