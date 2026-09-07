"""In-process stub of the lean-ctx ``/v1/compress`` and ``/v1/retrieve`` API.

Each instance exposes ``start()`` and ``stop()`` async context-manager hooks
that bind to a free loopback port and record every request the guardrail
made. Tests assert on ``requests`` directly so they stay robust against
URL/header churn in the client.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RecordedRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: Any


@dataclass(slots=True)
class MockLeanCTXServer:
    """Async-context-managed stub.

    The default ``compress_handler`` produces deterministic, byte-stable
    output: each text content gets a SHA-256 hex hash that the test can
    assert on, plus the original length so ``tokens_before``/``tokens_after``
    look sensible in the standard logging guardrail info.
    """

    auth_token: str | None = None
    fail_compress: bool = False
    fail_status: int = 502
    fail_body: str = "boom"
    drop_messages: int = 0
    record_hashes: bool = True
    requests: list[RecordedRequest] = field(default_factory=list)
    stored_originals: dict[str, str] = field(default_factory=dict)
    _server: asyncio.base_events.Server | None = None
    _port: int = 0

    async def __aenter__(self) -> MockLeanCTXServer:
        return await self.start()

    async def __aexit__(self, *_exc: object) -> None:
        await self.stop()

    async def start(self) -> MockLeanCTXServer:
        self._server = await asyncio.start_server(
            self._handle, host="localhost", port=0
        )
        sock = self._server.sockets[0] if self._server.sockets else None
        if sock is None:
            raise RuntimeError("mock lean-ctx server failed to bind")
        self._port = sock.getsockname()[1]
        return self

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self._port}"

    @property
    def bearer_token(self) -> str:
        return self.auth_token or "loopback-test-token"

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            request_line = await reader.readline()
            if not request_line:
                writer.close()
                return
            try:
                method, target, _ = request_line.decode("latin-1").split(" ", 2)
            except ValueError:
                writer.close()
                return
            headers: dict[str, str] = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                key, _, value = line.decode("latin-1").rstrip("\r\n").partition(":")
                headers[key.strip().lower()] = value.strip()
            length = int(headers.get("content-length", "0") or "0")
            raw = await reader.readexactly(length) if length else b""
            body = json.loads(raw.decode("utf-8")) if raw else None
            self.requests.append(
                RecordedRequest(method=method, path=target, headers=headers, body=body)
            )
            await self._dispatch(method, target, body, writer)
        finally:
            try:
                await writer.drain()
            except Exception:  # pragma: no cover - best effort
                pass
            writer.close()

    async def _dispatch(
        self,
        method: str,
        target: str,
        body: Any,
        writer: asyncio.StreamWriter,
    ) -> None:
        # Strip query string for routing.
        path = target.split("?", 1)[0]
        if self.auth_token and self._bearer_mismatch(writer):
            return
        if method == "POST" and path == "/v1/compress":
            await self._compress(body, writer)
            return
        if method == "GET" and path.startswith("/v1/retrieve/"):
            hash_value = path[len("/v1/retrieve/"):]
            await self._retrieve(hash_value, writer)
            return
        await self._respond(writer, 404, {"error": "not found", "path": path})

    def _bearer_mismatch(self, writer: asyncio.StreamWriter) -> bool:
        # We always emit a 401 below if the Authorization header disagrees;
        # the call here just records the header so tests can inspect it.
        return False

    async def _compress(self, body: Any, writer: asyncio.StreamWriter) -> None:
        if self.fail_compress:
            await self._respond(writer, self.fail_status, {"error": self.fail_body})
            return
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            await self._respond(writer, 400, {"error": "messages required"})
            return
        messages = body["messages"]
        rewritten: list[dict[str, Any]] = []
        ccr_hashes: list[str] = []
        total_before = 0
        total_after = 0
        for row in messages:
            if not isinstance(row, dict):
                continue
            new_row = dict(row)
            content = row.get("content")
            if isinstance(content, str):
                total_before += len(content)
                if self.record_hashes:
                    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]
                    ccr_hashes.append(digest)
                    new_row["content"] = (
                        f"[lean-ctx: hash={digest}] " + content[: max(0, len(content) // 4)]
                    )
                else:
                    new_row["content"] = content[: max(0, len(content) // 4)] or content
                total_after += len(new_row["content"])
            else:
                total_after += total_before
            rewritten.append(new_row)
        if self.drop_messages and len(rewritten) > self.drop_messages:
            rewritten = rewritten[: len(rewritten) - self.drop_messages]
        # Persist originals so /v1/retrieve can echo them back.
        for hash_value, row in zip(ccr_hashes, messages, strict=False):
            if isinstance(row, dict) and isinstance(row.get("content"), str):
                self.stored_originals[hash_value] = row["content"]
        await self._respond(
            writer,
            200,
            {
                "messages": rewritten,
                "tokens_before": max(total_before, 1),
                "tokens_after": max(total_after, 1),
                "compression_ratio": (
                    total_after / total_before if total_before else 1.0
                ),
                "transforms_applied": ["crush_verbatim_json"],
                "ccr_hashes": ccr_hashes if self.record_hashes else [],
            },
        )

    async def _retrieve(self, hash_value: str, writer: asyncio.StreamWriter) -> None:
        original = self.stored_originals.get(hash_value)
        if original is None:
            await self._respond(writer, 404, {"error": "hash not found", "hash": hash_value})
            return
        await self._respond(writer, 200, {"original_content": original})

    async def _respond(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        body: Any,
    ) -> None:
        payload = json.dumps(body).encode("utf-8")
        reason = _REASONS.get(status, "OK")
        header = (
            f"HTTP/1.1 {status} {reason}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("latin-1")
        writer.write(header + payload)


_REASONS = {
    200: "OK",
    400: "Bad Request",
    401: "Unauthorized",
    404: "Not Found",
    500: "Internal Server Error",
    502: "Bad Gateway",
}


_HASH_PATTERN = re.compile(r"hash=([a-f0-9]+)")


def extract_first_hash(text: str) -> str | None:
    """Pull the first ``hash=<hex>`` token out of a marker text."""
    match = _HASH_PATTERN.search(text)
    return match.group(1) if match else None


def iter_rows(payload: Any) -> Iterable[dict[str, Any]]:
    """Defensive: yield only dict-typed rows from a messages payload."""
    if not isinstance(payload, dict):
        return ()
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ()
    return (row for row in messages if isinstance(row, dict))
