{
  description = "litellm-guardrail-lean-ctx development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python313;
        pythonPackages = pkgs.python313Packages;
      in
      {
        devShells.default = pkgs.mkShell {
          name = "litellm-guardrail-lean-ctx";
          # Keep the toolchain minimal: Python 3.13 + the test runner, the
          # linter/formatter, and the type checker. `uv sync` pulls the
          # actual project deps (litellm, httpx, pytest-asyncio, pyyaml) from
          # pyproject.
          packages = with pkgs; [
            python
            uv
            pythonPackages.pytest
            pythonPackages.pytest-asyncio
            pythonPackages.ruff
            pythonPackages.ty
            stdenv.cc.cc.lib
          ];

          shellHook = ''
            export PYTHONDONTWRITEBYTECODE=1
            export PYTHONUNBUFFERED=1
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          '';

          UV_PYTHON = "${python}/bin/python3";
        };

        apps = {
          lint = {
            type = "app";
            program = "${pythonPackages.ruff}/bin/ruff";
            args = [ "check" "src" "tests" ];
          };
          format = {
            type = "app";
            program = "${pythonPackages.ruff}/bin/ruff";
            args = [ "format" "src" "tests" ];
          };
          ty = {
            type = "app";
            program = "${pythonPackages.ty}/bin/ty";
            args = [ "check" "src" ];
          };
        };
      });
}
