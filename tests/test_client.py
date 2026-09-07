"""Tests for the lean-ctx HTTP client (stdlib-only transport)."""

from __future__ import annotations

import pytest

from litellm_guardrail_lean_ctx.client import (
    DEFAULT_TIMEOUT_SECONDS,
    LeanCTXClient,
    LeanCTXHashMissing,
    LeanCTXHTTPError,
    LeanCTXUnreachable,
)

from .mock_server import MockLeanCTXServer, extract_first_hash

pytestmark = pytest.mark.asyncio


async def test_compress_round_trip(mock_server: MockLeanCTXServer) -> None:
    client = LeanCTXClient(mock_server.base_url, api_key=mock_server.bearer_token)
    messages = [
        {"role": "system", "content": "you are lean-ctx"},
        {"role": "user", "content": "hi there friend"},
        {"role": "assistant", "content": "hello!"},
    ]
    result = await client.compress(messages, model="gpt-4o")
    assert len(result.messages) == len(messages)
    assert result.stats.tokens_before is not None and result.stats.tokens_before > 0
    assert result.stats.tokens_after is not None
    assert result.stats.compression_ratio is not None
    assert result.ccr_hashes, "expected the mock to issue at least one hash"
    # Hashes are stable across calls for the same input content.
    user_hash = extract_first_hash(result.messages[1]["content"])
    assert user_hash in result.ccr_hashes


async def test_compress_sends_model_and_bearer(mock_server: MockLeanCTXServer) -> None:
    client = LeanCTXClient(mock_server.base_url, api_key="secret-token-12345")
    await client.compress(
        [{"role": "user", "content": "ping"}],
        model="claude-sonnet-4",
    )
    request = mock_server.requests[-1]
    assert request.method == "POST"
    assert request.path == "/v1/compress"
    assert request.headers["authorization"] == "Bearer secret-token-12345"
    assert request.body["model"] == "claude-sonnet-4"


async def test_compress_rejects_row_count_drift() -> None:
    server = MockLeanCTXServer(drop_messages=1)
    await server.start()
    try:
        client = LeanCTXClient(server.base_url)
        with pytest.raises(LeanCTXHTTPError) as exc_info:
            await client.compress(
                [
                    {"role": "user", "content": "a"},
                    {"role": "user", "content": "b"},
                ],
            )
        assert exc_info.value.status_code == 200
        assert "row count" in exc_info.value.body or "changed the message" in str(exc_info.value)
    finally:
        await server.stop()


async def test_compress_unreachable_maps_to_typed_error() -> None:
    client = LeanCTXClient("http://localhost:1")  # unbound port
    with pytest.raises(LeanCTXUnreachable):
        await client.compress([{"role": "user", "content": "hi"}])


async def test_compress_http_error_carries_status() -> None:
    server = MockLeanCTXServer(fail_compress=True, fail_status=502, fail_body="upstream down")
    await server.start()
    try:
        client = LeanCTXClient(server.base_url)
        with pytest.raises(LeanCTXHTTPError) as exc_info:
            await client.compress([{"role": "user", "content": "x"}])
        assert exc_info.value.status_code == 502
        assert "upstream down" in exc_info.value.body
    finally:
        await server.stop()


async def test_retrieve_returns_original_content(mock_server: MockLeanCTXServer) -> None:
    client = LeanCTXClient(mock_server.base_url)
    compressed = await client.compress([{"role": "user", "content": "expand me please"}])
    hash_value = next(iter(compressed.ccr_hashes))
    original = await client.retrieve(hash_value)
    assert original == "expand me please"


async def test_retrieve_missing_hash_raises_typed_error(mock_server: MockLeanCTXServer) -> None:
    client = LeanCTXClient(mock_server.base_url)
    with pytest.raises(LeanCTXHashMissing):
        await client.retrieve("deadbeef")


async def test_invalid_timeout_rejected() -> None:
    with pytest.raises(ValueError):
        LeanCTXClient("http://localhost:1", timeout=0)
    with pytest.raises(ValueError):
        LeanCTXClient("http://localhost:1", timeout=-1)
    with pytest.raises(ValueError):
        LeanCTXClient("http://localhost:1", timeout=float("inf"))


async def test_default_timeout_is_documented() -> None:
    assert DEFAULT_TIMEOUT_SECONDS > 0
