"""Shared fixtures: a mock lean-ctx server and a wired-up guardrail instance.

The guardrail imports ``litellm``, which is only available in the test
environment. We construct the guardrail manually here so the rest of the
test suite can stay focused on behaviour.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from litellm_guardrail_lean_ctx import LeanCTXGuardrail
from litellm_guardrail_lean_ctx.guardrail import GUARDRAIL_PROVIDER_NAME

from .mock_server import MockLeanCTXServer


@pytest.fixture
async def mock_server() -> AsyncIterator[MockLeanCTXServer]:
    server = MockLeanCTXServer()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


@pytest.fixture
async def guardrail(mock_server: MockLeanCTXServer) -> LeanCTXGuardrail:
    return LeanCTXGuardrail(
        api_base=mock_server.base_url,
        api_key=mock_server.bearer_token,
        guardrail_name="lean-ctx-test",
        event_hook="pre_call",
        default_on=True,
    )


@pytest.fixture
async def fail_open_guardrail(mock_server: MockLeanCTXServer) -> LeanCTXGuardrail:
    return LeanCTXGuardrail(
        api_base=mock_server.base_url,
        api_key=mock_server.bearer_token,
        guardrail_name="lean-ctx-test",
        event_hook="pre_call",
        default_on=True,
        unreachable_fallback="fail_open",
    )


def make_inputs(messages: list[dict]) -> dict:
    """Build the ``GenericGuardrailAPIInputs`` shape the guardrail consumes."""
    return {"structured_messages": list(messages), "texts": [], "images": []}


def assert_provider(recorded: dict) -> None:
    """Confirm the guardrail records its provider in standard logging info."""
    assert recorded.get("guardrail_provider") == GUARDRAIL_PROVIDER_NAME


__all__ = [
    "MockLeanCTXServer",
    "assert_provider",
    "fail_open_guardrail",
    "guardrail",
    "make_inputs",
]
