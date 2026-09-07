"""End-to-end tests for ``LeanCTXGuardrail.apply_guardrail``."""

from __future__ import annotations

import pytest

from litellm_guardrail_lean_ctx import LEAN_CTX_RETRIEVE_TOOL_NAME
from litellm_guardrail_lean_ctx.guardrail import LeanCTXGatewayError

from .conftest import make_inputs
from .mock_server import MockLeanCTXServer

pytestmark = pytest.mark.asyncio


_BASE_REQUEST = {
    "model": "gpt-4o",
    "proxy_server_request": {"headers": {}},
}


def _request_with_headers(headers: dict[str, str]) -> dict:
    base = dict(_BASE_REQUEST)
    base["proxy_server_request"] = {"headers": dict(headers)}
    return base


async def test_apply_guardrail_compresses_and_records_stats(
    guardrail, mock_server: MockLeanCTXServer
) -> None:
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "first user turn"},
        {"role": "assistant", "content": "first assistant"},
        {"role": "user", "content": "second user turn"},
    ]
    inputs = make_inputs(messages)
    request_data = dict(_BASE_REQUEST)

    result = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data=request_data,
        input_type="request",
    )
    rewritten = result["structured_messages"]
    # System + last user rows are protected, so they remain byte-identical.
    assert rewritten[0] == messages[0]
    assert rewritten[-1] == messages[-1]
    # The middle two rows are compressible.
    assert rewritten[1]["role"] == "user"
    assert "lean-ctx" in rewritten[1]["content"]
    # CCR retrieval tool is injected because the mock emits hashes.
    tools = result["tools"]
    assert any(t["function"]["name"] == LEAN_CTX_RETRIEVE_TOOL_NAME for t in tools)
    # The mock saw exactly one /v1/compress call.
    assert len(mock_server.requests) == 1
    assert mock_server.requests[0].path == "/v1/compress"
    assert mock_server.requests[0].headers["authorization"] == f"Bearer {mock_server.bearer_token}"


async def test_apply_guardrail_skips_background_requests(guardrail) -> None:
    messages = [{"role": "user", "content": "hi"}]
    request_data = dict(_BASE_REQUEST)
    request_data["background"] = True
    result = await guardrail.apply_guardrail(
        inputs=make_inputs(messages),
        request_data=request_data,
        input_type="request",
    )
    assert result["structured_messages"] == messages


async def test_apply_guardrail_skips_response_input_type(guardrail) -> None:
    messages = [{"role": "user", "content": "hi"}]
    result = await guardrail.apply_guardrail(
        inputs=make_inputs(messages),
        request_data=dict(_BASE_REQUEST),
        input_type="response",
    )
    assert result["structured_messages"] == messages


async def test_apply_guardrail_honours_bypass_header(guardrail) -> None:
    messages = [{"role": "user", "content": "hi"}]
    request_data = _request_with_headers({"x-lean-ctx-bypass": "true"})
    result = await guardrail.apply_guardrail(
        inputs=make_inputs(messages),
        request_data=request_data,
        input_type="request",
    )
    assert result["structured_messages"] == messages


async def test_apply_guardrail_returns_inputs_on_empty_messages(guardrail) -> None:
    inputs = {"structured_messages": [], "texts": [], "images": []}
    result = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data=dict(_BASE_REQUEST),
        input_type="request",
    )
    assert result is inputs


async def test_apply_guardrail_protects_last_user_even_when_compressible(guardrail) -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "earlier"},
        {"role": "user", "content": "instruction"},
    ]
    result = await guardrail.apply_guardrail(
        inputs=make_inputs(messages),
        request_data=dict(_BASE_REQUEST),
        input_type="request",
    )
    rewritten = result["structured_messages"]
    # The instruction is the last user message and must remain intact.
    assert rewritten[-1]["content"] == "instruction"


async def test_apply_guardrail_passes_non_text_parts_through(guardrail) -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this image"},
                {"type": "image_url", "image_url": {"url": "localhost"}},
            ],
        },
    ]
    result = await guardrail.apply_guardrail(
        inputs=make_inputs(messages),
        request_data=dict(_BASE_REQUEST),
        input_type="request",
    )
    rewritten = result["structured_messages"]
    # The image-bearing row passes through untouched (text + image_url preserved).
    assert rewritten[1]["content"] == messages[1]["content"]


async def test_apply_guardrail_fail_closed_raises(guardrail, mock_server) -> None:
    mock_server.fail_compress = True
    mock_server.fail_status = 502
    mock_server.fail_body = "upstream down"
    try:
        with pytest.raises(LeanCTXGatewayError) as exc_info:
            await guardrail.apply_guardrail(
                inputs=make_inputs([
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "earlier"},
                    {"role": "user", "content": "hi"},
                ]),
                request_data=dict(_BASE_REQUEST),
                input_type="request",
            )
        assert exc_info.value.detail["status_code"] == 502
    finally:
        mock_server.fail_compress = False


async def test_apply_guardrail_fail_open_forwards_originals(fail_open_guardrail) -> None:
    # Sanity-check attribute access by binding the client locally.
    _ = fail_open_guardrail._client
    # We need the mock server reachable but failing.
    # The fixture already bound the client to the mock; we point it at a
    # separate failing server to exercise the failure path cleanly.
    from litellm_guardrail_lean_ctx.client import LeanCTXClient

    failing_server = MockLeanCTXServer(fail_compress=True, fail_status=503, fail_body="nope")
    await failing_server.start()
    try:
        fail_open_guardrail._client = LeanCTXClient(failing_server.base_url)
        messages = [{"role": "user", "content": "keep me intact"}]
        result = await fail_open_guardrail.apply_guardrail(
            inputs=make_inputs(messages),
            request_data=dict(_BASE_REQUEST),
            input_type="request",
        )
        # fail_open returns the original messages uncompressed.
        assert result["structured_messages"] == messages
    finally:
        await failing_server.stop()


async def test_apply_guardrail_ccr_disabled(guardrail) -> None:
    guardrail.ccr_retrieval = False
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "expand me"},
        {"role": "user", "content": "instruction"},
    ]
    result = await guardrail.apply_guardrail(
        inputs=make_inputs(messages),
        request_data=dict(_BASE_REQUEST),
        input_type="request",
    )
    # CCR off: no retrieve tool should be injected even though the mock
    # would have emitted hashes.
    assert "tools" not in result


async def test_apply_guardrail_protects_just_retrieved_tool_results(
    guardrail, mock_server: MockLeanCTXServer
) -> None:
    """A tool result carrying a CCR expansion must not be re-compressed.

    The retrieve tool call + its tool result form a single tool exchange,
    which the guardrail expands wholesale into the protected set so neither
    row is sent to ``/v1/compress`` again.
    """
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "earlier turn to compress"},
        {"role": "user", "content": "instruction"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lean_ctx_retrieve", "arguments": '{"hash": "abc"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "verbatim expanded text"},
    ]
    request_data = dict(_BASE_REQUEST)
    # The untranslated request also carries the same tool_call, so the
    # guardrail can pair the tool result back to the retrieve call.
    request_data["messages"] = messages
    result = await guardrail.apply_guardrail(
        inputs=make_inputs(messages),
        request_data=request_data,
        input_type="request",
    )
    rewritten = result["structured_messages"]
    # Both the assistant tool call row and its tool result row are protected
    # verbatim because the retrieve exchange is held back wholesale.
    tool_row = next(r for r in rewritten if r.get("role") == "tool")
    assistant_row = next(r for r in rewritten if r.get("role") == "assistant")
    assert tool_row["content"] == "verbatim expanded text"
    assert assistant_row["tool_calls"][0]["id"] == "call-1"
    # The earlier user turn is compressible, so a single /v1/compress call is made.
    assert len(mock_server.requests) == 1
    assert mock_server.requests[0].path == "/v1/compress"
