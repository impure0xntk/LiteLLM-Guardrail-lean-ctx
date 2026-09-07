"""Tests for the config model and registry wiring."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from litellm_guardrail_lean_ctx import (
    LeanCTXGuardrail,
    LeanCTXGuardrailConfigModel,
    guardrail_class_registry,
    guardrail_initializer_registry,
    initialize_guardrail,
    register,
)


def test_config_model_defaults() -> None:
    cfg = LeanCTXGuardrailConfigModel()
    assert cfg.api_base is None
    assert cfg.unreachable_fallback == "fail_closed"
    assert cfg.ccr_retrieval is True


def test_config_model_coerces_timeout() -> None:
    cfg = LeanCTXGuardrailConfigModel(timeout="12.5")
    assert cfg.timeout == 12.5


def test_config_model_rejects_non_numeric_timeout() -> None:
    with pytest.raises(ValidationError):
        LeanCTXGuardrailConfigModel(timeout="soon")


def test_config_model_rejects_unknown_fallback() -> None:
    with pytest.raises(ValidationError):
        LeanCTXGuardrailConfigModel(unreachable_fallback="maybe")  # type: ignore[arg-type]


def test_config_model_ui_friendly_name() -> None:
    assert LeanCTXGuardrailConfigModel.ui_friendly_name() == "LeanCTX"


def test_registry_keys_present() -> None:
    assert "lean-ctx" in guardrail_class_registry
    assert "lean-ctx" in guardrail_initializer_registry
    assert guardrail_class_registry["lean-ctx"] is LeanCTXGuardrail
    assert guardrail_initializer_registry["lean-ctx"] is initialize_guardrail


def test_register_updates_litellm_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    class_registry = getattr(litellm, "guardrail_class_registry", None)
    initializer_registry = getattr(litellm, "guardrail_initializer_registry", None)
    if class_registry is None:
        class_registry = {}
        litellm.guardrail_class_registry = class_registry
    if initializer_registry is None:
        initializer_registry = {}
        litellm.guardrail_initializer_registry = initializer_registry
    # Wipe any prior state for the key so the assertion below is meaningful.
    class_registry.pop("lean-ctx", None)
    initializer_registry.pop("lean-ctx", None)
    register()
    assert class_registry["lean-ctx"] is LeanCTXGuardrail
    assert initializer_registry["lean-ctx"] is initialize_guardrail
