"""Async HTTP client for the lean-ctx ``/v1/compress`` and ``/v1/retrieve`` endpoints.

Built on ``asyncio`` + ``http.client`` so the runtime cost is a handful of bytes
and no third-party HTTP library is pulled in alongside ``litellm``.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 60.0
CONNECT_TIMEOUT_SECONDS = 10.0
_HASH_PATTERN_FRAGMENT = "[a-f0-9]{12,24}"


class LeanCTXHTTPError(RuntimeError):
    """Raised when the lean-ctx server returns an unexpected response.

    Carries the upstream status code and body so the guardrail can surface a
    useful error to the litellm-proxy admin UI without losing the original
    payload for debugging.
    """

    def __init__(self, message: str, *, status_code: int, body: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class LeanCTXUnreachable(RuntimeError):
    """Raised when the lean-ctx server cannot be reached at all."""


@dataclass(slots=True)
class CompressStats:
    """Token accounting returned by the lean-ctx compression service."""

    tokens_before: int | None = None
    tokens_after: int | None = None
    tokens_saved: int | None = None
    compression_ratio: float | None = None
    transforms_applied: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.tokens_before is not None:
            out["tokens_before"] = self.tokens_before
        if self.tokens_after is not None:
            out["tokens_after"] = self.tokens_after
        if self.tokens_saved is not None:
            out["tokens_saved"] = self.tokens_saved
        elif self.tokens_before is not None and self.tokens_after is not None:
            # Spend tracking reads tokens_saved; derive it when the server omits
            # the field so savings are still counted.
            out["tokens_saved"] = self.tokens_before - self.tokens_after
        if self.compression_ratio is not None:
            out["compression_ratio"] = self.compression_ratio
        if self.transforms_applied:
            out["transforms_applied"] = list(self.transforms_applied)
        return out


@dataclass(slots=True)
class CompressResult:
    """Result of a ``/v1/compress`` call."""

    messages: list[dict[str, Any]]
    stats: CompressStats
    ccr_hashes: frozenset[str]


@dataclass(slots=True)
class _Endpoint:
    """Parsed ``api_base`` ready to feed to ``http.client``."""

    host: str
    port: int
    use_tls: bool

    @classmethod
    def parse(cls, api_base: str) -> _Endpoint:
        parsed = urllib.parse.urlparse(api_base.rstrip("/"))
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"Unsupported scheme {parsed.scheme!r}; expected http or https."
            )
        host = parsed.hostname or ""
        if not host:
            raise ValueError(f"api_base {api_base!r} has no host component.")
        # Default ports mirror urllib.parse; explicit ports in the URL win.
        if parsed.port is not None:
            port = parsed.port
        elif parsed.scheme == "https":
            port = 443
        else:
            port = 80
        return cls(host=host, port=port, use_tls=parsed.scheme == "https")


class LeanCTXClient:
    """Async wrapper around the lean-ctx ``/v1`` contract."""

    def __init__(
        self,
        api_base: str,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._endpoint = _Endpoint.parse(api_base)
        self._api_key = api_key
        # The transport accepts zero / negative / non-finite values silently
        # (zero == no deadline, inf == never times out, negative == past).
        # Validate up front so a bad config surfaces immediately rather than
        # as a hung request.
        import math
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
            raise ValueError(
                f"timeout must be a positive number of seconds, got {timeout!r}."
            )
        value = float(timeout)
        if not (value > 0 and math.isfinite(value)):
            raise ValueError(
                f"timeout must be a positive finite number of seconds, got {timeout!r}."
            )
        self._timeout = value

    @property
    def api_base(self) -> str:
        scheme = "https" if self._endpoint.use_tls else "http"
        return f"{scheme}://{self._endpoint.host}:{self._endpoint.port}"

    def _request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def compress(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
    ) -> CompressResult:
        """Call ``POST /v1/compress`` and return the rewritten messages + stats.

        The lean-ctx contract is ``{messages: [...]}`` in, ``{messages: [...],
        tokens_before, tokens_after, compression_ratio, ...}`` out. Row count
        must be preserved; a service that drops or duplicates rows is treated
        as a hard failure because the protected rows we held back cannot be
        re-interleaved into a reshaped conversation.
        """
        payload: dict[str, Any] = {"messages": messages}
        if model:
            payload["model"] = model
        body = await self._request(
            method="POST",
            path="/v1/compress",
            payload=payload,
        )
        return self._parse_compress_response(body, expected_rows=len(messages))

    async def retrieve(self, hash_value: str, *, query: str | None = None) -> str:
        """Call ``GET /v1/retrieve/{hash}`` and return the original content.

        A 404 is the documented response for an expired or unknown hash; the
        caller may want to render it as a marker rather than abort the request,
        so we surface it as a typed exception.
        """
        path = f"/v1/retrieve/{urllib.parse.quote(hash_value, safe='')}"
        params: list[tuple[str, str]] = []
        if query:
            params.append(("query", query))
        try:
            body = await self._request(method="GET", path=path, params=params)
        except LeanCTXHTTPError as exc:
            if exc.status_code == 404:
                raise LeanCTXHashMissing(hash_value) from exc
            raise
        if isinstance(body, Mapping):
            original = body.get("original_content")
            if isinstance(original, str):
                return original
        # Bare-string bodies are also accepted; lean-ctx returns the original
        # text directly when JSON encoding would not add information.
        return body if isinstance(body, str) else json.dumps(body)

    def _parse_compress_response(
        self,
        body: Any,
        *,
        expected_rows: int,
    ) -> CompressResult:
        if not isinstance(body, Mapping):
            raise LeanCTXHTTPError(
                "lean-ctx /v1/compress returned non-object body",
                status_code=200,
                body=str(body)[:500],
            )
        raw_messages = body.get("messages")
        if not isinstance(raw_messages, list):
            raise LeanCTXHTTPError(
                "lean-ctx /v1/compress response missing 'messages'",
                status_code=200,
                body=str(body)[:500],
            )
        compressed = [row for row in raw_messages if isinstance(row, dict)]
        if not compressed:
            raise LeanCTXHTTPError(
                "lean-ctx /v1/compress returned an empty message list",
                status_code=200,
                body=str(body)[:500],
            )
        if len(compressed) != expected_rows:
            # Rows are matched positionally when the never-compressed messages
            # are put back, so a reshaped conversation cannot be applied at all.
            raise LeanCTXHTTPError(
                "lean-ctx /v1/compress changed the message count",
                status_code=200,
                body=f"sent={expected_rows} returned={len(compressed)}",
            )

        stats = CompressStats(
            tokens_before=_coerce_int(body.get("tokens_before")),
            tokens_after=_coerce_int(body.get("tokens_after")),
            tokens_saved=_coerce_int(body.get("tokens_saved")),
            compression_ratio=_coerce_float(body.get("compression_ratio")),
            transforms_applied=_coerce_str_list(body.get("transforms_applied")),
        )

        return CompressResult(
            messages=compressed,
            stats=stats,
            ccr_hashes=_read_ccr_hashes(body.get("ccr_hashes")),
        )

    async def _request(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        params: list[tuple[str, str]] | None = None,
    ) -> Any:
        query = urllib.parse.urlencode(params) if params else ""
        full_path = f"{path}?{query}" if query else path
        body_bytes = json.dumps(payload).encode("utf-8") if payload is not None else b""
        headers = self._request_headers()
        if payload is not None:
            headers["Content-Length"] = str(len(body_bytes))

        # http.client is sync; the only way to interleave with the litellm event
        # loop is to run the blocking call in the default thread pool. This keeps
        # the guardrail concurrent-safe without pulling in httpx / aiohttp.
        try:
            raw_status, raw_body = await asyncio.to_thread(
                _sync_request,
                self._endpoint,
                full_path,
                method,
                headers,
                body_bytes,
                self._timeout,
            )
        except (TimeoutError, OSError) as exc:
            raise LeanCTXUnreachable(str(exc)) from exc

        if raw_status >= 400:
            raise LeanCTXHTTPError(
                f"lean-ctx responded {raw_status}",
                status_code=raw_status,
                body=raw_body,
            )
        if not raw_body:
            return {}
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError:
            return raw_body


class LeanCTXHashMissing(LeanCTXHTTPError):
    """Specialised 404 used by ``retrieve`` for expired or unknown hashes."""

    def __init__(self, hash_value: str) -> None:
        super().__init__(
            f"lean-ctx has no record for hash={hash_value}",
            status_code=404,
            body="",
        )
        self.hash_value = hash_value


def _sync_request(
    endpoint: _Endpoint,
    path: str,
    method: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout: float,
) -> tuple[int, str]:
    """Blocking single-request transport; runs in a thread pool."""
    connect_timeout = min(timeout, CONNECT_TIMEOUT_SECONDS)
    if endpoint.use_tls:
        connection: http.client.HTTPConnection = http.client.HTTPSConnection(
            endpoint.host,
            endpoint.port,
            timeout=connect_timeout,
        )
    else:
        connection = http.client.HTTPConnection(
            endpoint.host,
            endpoint.port,
            timeout=connect_timeout,
        )
    try:
        connection.request(method, path, body=body, headers=dict(headers))
        response = connection.getresponse()
        status = response.status
        raw = response.read()
    finally:
        connection.close()
    return status, raw.decode("utf-8", errors="replace")


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _read_ccr_hashes(value: Any) -> frozenset[str]:
    """Extract the lowercase CCR hashes the compression service reported."""
    if not isinstance(value, list):
        return frozenset()
    import re

    pattern = re.compile(_HASH_PATTERN_FRAGMENT)
    out: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        candidate = item.lower()
        if pattern.fullmatch(candidate):
            out.add(candidate)
    return frozenset(out)
