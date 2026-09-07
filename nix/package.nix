{
  # Derivation for `litellm-guardrail-lean-ctx`.
  #
  # Mirrors pyproject.toml: hatchling-built wheel with `litellm>=1.50`
  # as the single runtime dependency. Kept here so flake.nix stays a
  # thin composition layer.
  lib,
  buildPythonPackage,
  hatchling,
  python313,
  litellm,
}:

buildPythonPackage rec {
  pname = "litellm-guardrail-lean-ctx";
  version = "0.1.0";

  # Reuse the source that flake.nix is evaluated against; no PyPI
  # upload exists yet, so a pypi src is not an option.
  src = lib.cleanSource ../.;

  pyproject = true;

  build-system = [ hatchling ];

  buildInputs = [ ];

  propagatedBuildInputs = [ litellm ];

  # The project ships only `src/litellm_guardrail_lean_ctx/*.py`; no
  # data files, scripts, or compiled extensions.
  dontConfigure = true;

  # Hatchling picks the wheel target from pyproject; the test suite
  # pulls optional dev deps (httpx, pyyaml, pytest) that are not part
  # of the published distribution, so skip it here.
  doCheck = false;

  pythonImportsCheck = [ "litellm_guardrail_lean_ctx" ];

  meta = {
    description = "LiteLLM custom guardrail that compresses context via a lean-ctx proxy server.";
    homepage = "https://github.com/impure0xntk/LiteLLM-Guardrail-lean-ctx";
    license = lib.licenses.asl20;
    platforms = lib.platforms.unix;
  };
}
