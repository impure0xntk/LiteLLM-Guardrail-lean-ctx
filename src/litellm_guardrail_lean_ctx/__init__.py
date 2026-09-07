"""LiteLLM custom guardrail that compresses chat context via a lean-ctx proxy.

The package ships a single public class, :class:`LeanCTXGuardrail`, which extends
``litellm.integrations.custom_guardrail.CustomGuardrail``. It mirrors the shape of
the upstream Headroom guardrail so an existing litellm-proxy config that uses
``custom_guardrail`` only has to swap the module path.
"""

from __future__ import annotations

from .messages import (
    BYPASS_HEADER,
    LEAN_CTX_RETRIEVE_TOOL_NAME,
)

__all__ = [
    "BYPASS_HEADER",
    "LEAN_CTX_RETRIEVE_TOOL_NAME",
    "LeanCTXGuardrail",
    "LeanCTXGuardrailConfigModel",
    "guardrail_class_registry",
    "guardrail_initializer_registry",
    "initialize_guardrail",
    "register",
]


def __getattr__(name: str):  # PEP 562 lazy attribute access.
    # Importing ``guardrail`` at module load pulls in ``litellm``; we delay
    # that until the consumer actually asks for the guardrail class or one
    # of the registry hooks so importing the package stays cheap.
    if name in {
        "LeanCTXGuardrail",
        "LeanCTXGuardrailConfigModel",
        "initialize_guardrail",
        "register",
        "guardrail_class_registry",
        "guardrail_initializer_registry",
    }:
        from . import guardrail as _guardrail

        value = getattr(_guardrail, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'litellm_guardrail_lean_ctx' has no attribute {name!r}")
