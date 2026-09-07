"""Pydantic config models for the lean-ctx guardrail.

Mirrors the upstream Headroom layout so the litellm-proxy UI accepts the same
fields under a different integration name.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# LitellmParams is a multi-inheritance mixin that already carries ``api_key``
# and ``api_base``; we only need the lean-ctx specific knobs here.
try:  # pragma: no cover - import path varies across litellm versions
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
except Exception:  # pragma: no cover - keep import lazy for type-only use
    GuardrailConfigModel = BaseModel  # type: ignore[assignment,misc]


class LeanCTXGuardrailConfigModel(GuardrailConfigModel):  # type: ignore[misc]
    """Configuration schema exposed to the litellm-proxy admin UI."""

    api_base: str | None = Field(
        default=None,
        description=(
            "Base URL for the lean-ctx compression service (e.g. http://localhost:4444). "
            "Falls back to the LEAN_CTX_API_BASE env var."
        ),
    )
    api_key: str | None = Field(
        default=None,
        description=(
            "Bearer token forwarded as Authorization. Falls back to the LEAN_CTX_API_KEY "
            "env var, then the loopback session_token lean-ctx proxy prints on start."
        ),
    )
    model: str | None = Field(
        default=None,
        description="Model name forwarded to the lean-ctx /v1/compress endpoint.",
    )
    unreachable_fallback: Literal["fail_closed", "fail_open"] = Field(
        default="fail_closed",
        description=(
            "Behavior when the lean-ctx compression service is unreachable or errors. "
            "'fail_closed' raises an error (default). 'fail_open' logs a warning and "
            "forwards the request uncompressed."
        ),
    )
    timeout: float | None = Field(
        default=None,
        description=(
            "Per-call HTTP timeout in seconds. Defaults to 60s. "
            "Zero, negative and non-finite values are ignored."
        ),
    )
    ccr_retrieval: bool = Field(
        default=True,
        description=(
            "When True (default), the guardrail injects the lean-ctx retrieve tool "
            "and round-trips compressed tool results back to the original content."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "LeanCTX"
