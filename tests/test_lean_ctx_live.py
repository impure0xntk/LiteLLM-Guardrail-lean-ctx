"""End-to-end smoke test against a real lean-ctx proxy.

Run with ``RUN_LIVE_LEAN_CTX=1 uv run pytest tests/test_lean_ctx_live.py -q``
after starting the proxy via ``lean-ctx proxy start`` and exporting
``LEAN_CTX_API_KEY`` with the printed token. The proxy is left running by
the test consumer; the assertion simply verifies the wire format we built
against matches what the server returns.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest

from litellm_guardrail_lean_ctx import LeanCTXGuardrail

from .conftest import make_inputs

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def live_guardrail() -> AsyncIterator[LeanCTXGuardrail]:
    api_base = os.environ.get("LEAN_CTX_API_BASE", "http://localhost:4444")
    api_key = os.environ.get("LEAN_CTX_API_KEY")
    if not api_key:
        pytest.skip("LEAN_CTX_API_KEY not set; skipping live lean-ctx smoke test")
    guardrail = LeanCTXGuardrail(
        api_base=api_base,
        api_key=api_key,
        guardrail_name="lean-ctx-live",
        event_hook="pre_call",
        default_on=True,
    )
    try:
        yield guardrail
    finally:
        # No sockets held by the guardrail; nothing to close.
        pass


async def test_apply_guardrail_against_live_lean_ctx(live_guardrail: LeanCTXGuardrail) -> None:
    messages = [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Earlier discussion about context compression."},
        {"role": "user", "content": "Summarise the previous turn please."},
    ]
    request_data = {"model": "gpt-4o", "proxy_server_request": {"headers": {}}}
    result = await live_guardrail.apply_guardrail(
        inputs=make_inputs(messages),
        request_data=request_data,
        input_type="request",
    )
    rewritten = result["structured_messages"]
    # System + last user rows are protected.
    assert rewritten[0] == messages[0]
    assert rewritten[-1] == messages[-1]
    # The middle row has been passed through the real proxy.
    assert rewritten[1]["role"] == "user"
    assert isinstance(rewritten[1]["content"], str)
