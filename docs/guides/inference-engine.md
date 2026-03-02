# Inference Engine

`InferenceEngine` is the high-level API that wraps model discovery, server startup, and inference calls. It is designed to be the simplest way to run GGUF inference with `llama-server`.

## Core responsibilities

- Locate or bootstrap `llama-server` binaries
- Download GGUF models (registry, HuggingFace, or local paths)
- Launch and manage the server process
- Provide `infer()` / `generate()` helpers
- Collect basic runtime metrics
- Optionally initialize OpenTelemetry instrumentation

## Creating an engine

```python
import llamatelemetry as lt

engine = lt.InferenceEngine(
    server_url="http://127.0.0.1:8090",
    enable_telemetry=False,
)
```

## Loading a model

```python
engine.load_model(
    "gemma-3-1b-Q4_K_M",
    auto_start=True,
    auto_configure=True,
)
```

Key parameters:

- `model_name_or_path`: registry name, local path, or `repo:filename` syntax
- `gpu_layers`: layers offloaded to GPU (auto if None)
- `ctx_size`: context length
- `auto_start`: start `llama-server` if not already running
- `interactive_download`: prompt before downloading

## Inference methods

- `infer(prompt, **kwargs)` — primary inference call
- `generate(prompt, **kwargs)` — alias to `infer`
- `infer_stream(prompt, **kwargs)` — stream tokens from server
- `batch_infer(prompts, **kwargs)` — batch inference

```python
result = engine.infer("What is CUDA?", max_tokens=64)
print(result.text)
```

## Metrics

The engine maintains simple in-process metrics:

- `requests`, `total_tokens`, `total_latency_ms`
- `latencies` array for individual calls

```python
metrics = engine.get_metrics()
print(metrics)
```

## Telemetry integration

Enable telemetry during initialization:

```python
engine = lt.InferenceEngine(
    enable_telemetry=True,
    telemetry_config={
        "service_name": "llamatelemetry-demo",
        "otlp_endpoint": "http://localhost:4317",
        "enable_llama_metrics": True,
    },
)
```

See [Telemetry and Observability](telemetry-observability.md) for details.

## Cleanup

```python
engine.unload_model()
```

## Related reference

- [Core API](../reference/core-api.md)
- [Server and Models](../reference/server-models.md)
