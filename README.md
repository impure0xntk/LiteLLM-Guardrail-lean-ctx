# litellm-guardrail-lean-ctx

A LiteLLM [custom guardrail](https://docs.litellm.ai/docs/proxy/guardrails/custom_guardrail)
that compresses chat context through a running
[lean-ctx proxy](https://leanctx.com/docs/journeys/advanced-and-integrations/).

The shape, configuration model, and event hooks mirror the upstream
[Headroom guardrail](https://github.com/BerriAI/litellm/tree/litellm_internal_staging/litellm/proxy/guardrails/guardrail_hooks/headroom)
so a `litellm.yaml` that already references `headroom` can swap in
`lean-ctx` by changing one integration key.

## What it does

On every `pre_call`, the guardrail:

1. Pulls the proxy's `structured_messages` (system, history, tool results).
2. Sends the compressible rows to `POST /v1/compress` on the lean-ctx server.
3. Re-interleaves the rewritten rows with the rows that were held back
   (system, last user, last assistant, just-retrieved tool results).
4. If the response carries CCR hashes, injects the `lean_ctx_retrieve`
   tool so the model can ask for an expansion on a later turn.

The lean-ctx side speaks the same `messages`-in/`messages`-out contract that
LiteLLM's compression guardrail contract defines; the proxy is described as
`LiteLLM wire compatible v3.9.0` in the [lean-ctx docs](https://leanctx.com/docs/concepts/proxy/).

## Install

```bash
# from this repo (development)
nix develop           # see flake.nix for the devShell
uv sync --all-extras

# or, published form
pip install litellm-guardrail-lean-ctx
```

Python 3.13+ is required; `litellm>=1.50` is the only runtime dependency.
HTTP transport is stdlib-only (`asyncio` + `http.client`), so there is no
second HTTP stack to pin or update.

## Wire it into litellm-proxy

`examples/config.example.yaml`:

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: lean-ctx-compression
    litellm_params:
      guardrail: litellm_guardrail_lean_ctx.LeanCTXGuardrail
      mode: pre_call
      api_base: http://localhost:4444
      api_key: os.environ/LEAN_CTX_API_KEY
      model: gpt-4o
      unreachable_fallback: fail_closed
      ccr_retrieval: true
```

Lean-ctx prints a loopback bearer token when `lean-ctx proxy enable` runs;
point `LEAN_CTX_API_KEY` at it and the guardrail is authenticated.

The dotted `guardrail:` value points the proxy at the
`LeanCTXGuardrail` class directly; LiteLLM resolves it through
`get_instance_fn`, so no extra registration step is needed as long as the
package is importable from the proxy's working directory (e.g. drop a
`litellm_guardrail_lean_ctx/` folder next to `config.yaml`, or install the
wheel into the proxy's Python environment).

For embedded proxies that already import the package, you can also call
`register()` once at boot to expose the short key `lean-ctx`:

```python
import litellm_guardrail_lean_ctx
litellm_guardrail_lean_ctx.register()
```

## Configuration

| key | default | meaning |
| --- | --- | --- |
| `api_base` | `LEAN_CTX_API_BASE` env var | Lean-ctx proxy origin (e.g. `http://localhost:4444`). |
| `api_key` | `LEAN_CTX_API_KEY` env var | Bearer token. The lean-ctx proxy prints its loopback `session_token` on `lean-ctx proxy enable`. |
| `model` | request's own model | Forwarded to `/v1/compress`; lean-ctx uses it for ranking and transform selection. |
| `unreachable_fallback` | `fail_closed` | `fail_closed` raises; `fail_open` forwards uncompressed with a warning. |
| `timeout` | `60` | Per-call HTTP timeout in seconds. |
| `ccr_retrieval` | `true` | Inject the `lean_ctx_retrieve` tool and round-trip CCR markers. |

## Development

```bash
nix develop                                # Python 3.13 + uv + ruff + pytest
uv run pytest -q                           # unit tests against a mock lean-ctx server
uv run ruff check                          # lint
```

The flake exposes a single `devShell.<system>` with `python313`,
`uv`, `ruff`, `pytest`, `litellm`, `httpx`, and `pyyaml` pre-wired.
See `flake.nix` for the exact list.

## Tests

## Live integration test

`tests/test_lean_ctx_live.py` exercises the guardrail against a real
`lean-ctx proxy start` instance. With the proxy up, export the loopback
bearer token (printed by `lean-ctx proxy token`) and run:

```bash
lean-ctx proxy start --port=4444 -d
export LEAN_CTX_API_KEY="$(lean-ctx proxy token)"
export LEAN_CTX_API_BASE="http://localhost:4444"
uv run pytest tests/test_lean_ctx_live.py -v
lean-ctx proxy stop
```

Without the env vars the test is skipped, so the same command stays safe
in CI.

## Tests

Tests run an in-process lean-ctx stub (`tests/mock_server.py`) that speaks
the documented `/v1/compress` and `/v1/retrieve/{hash}` contract. They cover:

* Pre-call compression of system + history + tool rows.
* Row-count drift detection (a service that reshapes the conversation fails closed).
* Protected-row preservation (system, last user, last assistant, just-retrieved).
* CCR round-trip via the agentic-loop hook (hash validation, follow-up shape).
* Fail-open vs fail-close on lean-ctx outages.
* `x-lean-ctx-bypass` header short-circuit.
* Background-request skip.

## Layout

```
src/litellm_guardrail_lean_ctx/
  __init__.py        # lazy attribute access; defers litellm import
  client.py          # async http.client-based /v1/compress + /v1/retrieve client
  config.py          # pydantic config model
  guardrail.py       # LeanCTXGuardrail + litellm-proxy wiring
  messages.py        # pure helpers: protected indices, content flatten/restore
tests/
  conftest.py        # shared fixtures (mock lean-ctx server, guardrail instance)
  mock_server.py     # minimal /v1/compress + /v1/retrieve/{hash} stub
  test_*.py          # per-area unit tests
examples/
  config.example.yaml
flake.nix            # devShell for Python 3.13 + litellm + uv
pyproject.toml
```

## License

Apache-2.0.
