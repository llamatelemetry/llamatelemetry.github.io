# Troubleshooting

## Server fails to start

Checklist:

1. Confirm model path exists.
2. Confirm `LLAMA_SERVER_PATH` is valid if set.
3. Run with `silent=False` to capture stderr.
4. Check port conflicts (`8090` by default in high-level engine).

## Model load cancelled unexpectedly

`load_model_smart(..., interactive=True)` can return `None` if user declines download prompt.

Use:

- `interactive_download=False`, or
- provide local path, or
- pre-download model.

## Telemetry not active

Ensure OpenTelemetry packages are installed:

- `opentelemetry-api`
- `opentelemetry-sdk`
- optional OTLP exporters

Then validate:

```python
from llamatelemetry.telemetry import is_otel_available
print(is_otel_available())
```

## Windows encoding errors

On some Windows consoles, non-ASCII output can trigger encoding errors.

Workarounds:

- Use UTF-8 enabled terminal/session.
- Prefer notebook/Linux runtime for full Kaggle-focused workflows.
- Suppress noisy output (`silent=True`) where possible.

## Triton kernels not available

`llamatelemetry.cuda.triton_kernels` depends on optional Triton runtime. If absent:

- `list_kernels()` can be empty.
- Use fallback paths or skip Triton-specific features.

## GPU capability issues

`ServerManager.start_server` performs compatibility checks when `gpu_layers > 0`.

If your GPU is unsupported:

- set `gpu_layers=0` for CPU mode, or
- use compatible GPU hardware, or
- override with `skip_gpu_check=True` (not recommended).

## Kaggle secret loading problems

Use:

```python
from llamatelemetry.kaggle import auto_load_secrets
print(auto_load_secrets(set_env=True))
```

Ensure secret keys in Kaggle match expected names (`HF_TOKEN`, Graphistry keys).

## Where to go next

- [Server guide](server-management.md)
- [Model guide](model-management.md)
- [API reference](../reference/index.md)
