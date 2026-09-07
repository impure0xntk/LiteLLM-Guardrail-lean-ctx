"""Tests for the agentic-loop (CCR retrieval) hook."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from litellm_guardrail_lean_ctx import LEAN_CTX_RETRIEVE_TOOL_NAME
from litellm_guardrail_lean_ctx.guardrail import LeanCTXGuardrail

from .mock_server import MockLeanCTXServer, extract_first_hash

pytestmark = pytest.mark.asyncio


class _ChatCompletionResponse:
    """Minimal stand-in for a litellm ModelResponse with tool_calls."""

    def __init__(self, tool_calls, content: str = "") -> None:
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ]


async def _seed_issued_hashes(guardrail: LeanCTXGuardrail, mock_server: MockLeanCTXServer) -> str:
    """Run one compress to learn a real hash issued by the mock."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "expand me please"},
        {"role": "user", "content": "instruction"},
    ]
    request_data = {"model": "gpt-4o", "proxy_server_request": {"headers": {}}}
    result = await guardrail.apply_guardrail(
        inputs={"structured_messages": list(messages), "texts": [], "images": []},
        request_data=request_data,
        input_type="request",
    )
    rewritten = result["structured_messages"]
    hash_value = extract_first_hash(rewritten[1]["content"])
    assert hash_value is not None
    # Stash the call id so the agentic loop hook can scope validation.
    guardrail._issued_hashes_by_call_id["call-xyz"] = (
        frozenset({hash_value}),
        9e9,
    )
    return hash_value


async def test_should_run_agentic_loop_only_when_retrieve_tool_called(
    guardrail, mock_server: MockLeanCTXServer
) -> None:
    await _seed_issued_hashes(guardrail, mock_server)
    response = _ChatCompletionResponse(
        tool_calls=[
            SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(
                    name=LEAN_CTX_RETRIEVE_TOOL_NAME,
                    arguments='{"hash": "abc"}',
                ),
            )
        ]
    )
    tools = [{"type": "function", "function": {"name": LEAN_CTX_RETRIEVE_TOOL_NAME}}]
    should_run, payload = await guardrail.async_should_run_agentic_loop(
        response=response,
        model="gpt-4o",
        messages=[],
        tools=tools,
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )
    assert should_run is True
    assert payload["tool_calls"][0]["name"] == LEAN_CTX_RETRIEVE_TOOL_NAME


async def test_should_run_agentic_loop_false_without_retrieve_tool(
    guardrail, mock_server: MockLeanCTXServer
) -> None:
    response = _ChatCompletionResponse(
        tool_calls=[
            SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(name="other_tool", arguments="{}"),
            )
        ]
    )
    should_run, payload = await guardrail.async_should_run_agentic_loop(
        response=response,
        model="gpt-4o",
        messages=[],
        tools=[{"type": "function", "function": {"name": "other_tool"}}],
        stream=False,
        custom_llm_provider="openai",
        kwargs={},
    )
    assert should_run is False
    assert payload == {}


async def test_agentic_loop_plan_echoes_tool_call_and_tool_result(
    guardrail, mock_server: MockLeanCTXServer
) -> None:
    hash_value = await _seed_issued_hashes(guardrail, mock_server)
    original_messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "instruction"},
    ]
    response = _ChatCompletionResponse(
        content="let me look that up",
        tool_calls=[
            SimpleNamespace(
                id="call-xyz",
                function=SimpleNamespace(
                    name=LEAN_CTX_RETRIEVE_TOOL_NAME,
                    arguments=f'{{"hash": "{hash_value}"}}',
                ),
            )
        ],
    )
    plan = await guardrail.async_build_agentic_loop_plan(
        tools={"tool_calls": [{"id": "call-xyz", "name": LEAN_CTX_RETRIEVE_TOOL_NAME, "arguments": {"hash": hash_value}}]},
        model="gpt-4o",
        messages=list(original_messages),
        response=response,
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={},
        logging_obj=None,
        stream=False,
        kwargs={"litellm_call_id": "call-xyz"},
    )
    follow_up = plan.request_patch.messages
    # Original messages preserved at the front.
    assert follow_up[: len(original_messages)] == original_messages
    # Assistant echo + tool result appended.
    assistant = follow_up[len(original_messages)]
    tool = follow_up[len(original_messages) + 1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "let me look that up"
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "call-xyz"
    assert tool["content"] == "expand me please"


async def test_agentic_loop_rejects_hash_not_issued_for_call(
    guardrail, mock_server: MockLeanCTXServer
) -> None:
    # Seed only a *different* hash to ensure validation is per-call.
    guardrail._issued_hashes_by_call_id["call-xyz"] = (
        frozenset({"different-hash-123456abcdef"}),
        9e9,
    )
    response = _ChatCompletionResponse(
        tool_calls=[
            SimpleNamespace(
                id="call-xyz",
                function=SimpleNamespace(
                    name=LEAN_CTX_RETRIEVE_TOOL_NAME,
                    arguments='{"hash": "forged-hash-123456abcdef"}',
                ),
            )
        ],
    )
    plan = await guardrail.async_build_agentic_loop_plan(
        tools={"tool_calls": [{"id": "call-xyz", "name": LEAN_CTX_RETRIEVE_TOOL_NAME, "arguments": {"hash": "forged-hash-123456abcdef"}}]},
        model="gpt-4o",
        messages=[],
        response=response,
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={},
        logging_obj=None,
        stream=False,
        kwargs={"litellm_call_id": "call-xyz"},
    )
    tool = plan.request_patch.messages[-1]
    assert "was not produced" in tool["content"]
