# Inference Engine

`InferenceEngine` is the high-level API that orchestrates the entire inference lifecycle: model discovery, server startup, inference execution, metrics collection, and telemetry instrumentation. It is the recommended entry point for most llamatelemetry workflows.

## Overview

The engine wraps three subsystems into a single coherent interface:

1. **Model Management** -- locates GGUF models from the built-in registry, local paths, or HuggingFace
2. **Server Lifecycle** -- bootstraps, starts, monitors, and stops the `llama-server` process
3. **Inference Execution** -- sends prompts, collects results, and records performance metrics

## Creating an Engine

```python
import llamatelemetry as lt

# Minimal -- defaults to http://127.0.0.1:8090, no telemetry
engine = lt.InferenceEngine()

# Explicit configuration
engine = lt.InferenceEngine(
    server_url="http://127.0.0.1:8090",
    enable_telemetry=False,
    telemetry_config=None,
)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `server_url` | `str` | `"http://127.0.0.1:8090"` | Base URL of the llama-server instance |
| `enable_telemetry` | `bool` | `False` | Initialize OpenTelemetry tracing and metrics |
| `telemetry_config` | `dict` or `None` | `None` | Configuration dict passed to `setup_telemetry()` |

!!! note "Port Convention"
    `InferenceEngine` and `ServerManager` both default to port **8090**. The lower-level `LlamaCppClient` defaults to **8090**. When combining them manually, ensure ports match.

## Using as a Context Manager

The recommended pattern is to use the engine as a context manager, which guarantees cleanup on exit:

```python
import llamatelemetry as lt

with lt.InferenceEngine(server_url="http://127.0.0.1:8090") as engine:
    engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)
    result = engine.infer("What is CUDA?", max_tokens=128)
    print(result.text)
# Server is stopped and resources are released automatically
```

Without a context manager, you must call `engine.unload_model()` manually when finished.

## Loading a Model

```python
engine.load_model(
    "gemma-3-1b-Q4_K_M",     # Registry name, local path, or repo:filename
    gpu_layers=None,           # None = auto-detect optimal layers
    ctx_size=None,             # None = use model default
    auto_start=True,           # Start llama-server if not already running
    auto_configure=True,       # Auto-detect GPU layers and context size
    n_parallel=1,              # Number of parallel inference slots
)
```

### Model Sources

The `model_name_or_path` parameter accepts three formats:

**1. Registry name** -- looks up the built-in `MODEL_REGISTRY` (22 models):

```python
engine.load_model("gemma-3-1b-Q4_K_M")
engine.load_model("llama-3.2-1b-Q4_K_M")
engine.load_model("phi-4-mini-Q4_K_M")
```

**2. Local filesystem path** -- loads a GGUF file directly:

```python
engine.load_model("/home/user/models/my-model.gguf")
```

**3. HuggingFace repo:filename** -- downloads from HuggingFace Hub:

```python
engine.load_model("bartowski/gemma-2-2b-it-GGUF:gemma-2-2b-it-Q4_K_M.gguf")
```

### Auto-Configuration

When `auto_configure=True` (the default), the engine inspects available VRAM and selects appropriate values for `gpu_layers` and `ctx_size`. On a Tesla T4 with 16 GB VRAM, a 1B parameter Q4_K_M model typically gets all layers offloaded to GPU with a 4096-token context.

### Load Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | `str` | required | Registry name, local path, or `repo:filename` |
| `gpu_layers` | `int` or `None` | `None` | GPU layers to offload; `None` = auto |
| `ctx_size` | `int` or `None` | `None` | Context window size; `None` = auto |
| `auto_start` | `bool` | `True` | Start llama-server if not running |
| `auto_configure` | `bool` | `True` | Auto-detect optimal GPU/context settings |
| `n_parallel` | `int` | `1` | Number of parallel slots on the server |
| `interactive_download` | `bool` | `True` | Prompt before downloading large models |

## Inference Methods

### Single Inference

The primary method for running inference:

```python
result = engine.infer(
    prompt="Explain GPU memory hierarchy.",
    max_tokens=128,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    seed=0,
)

print(f"Text: {result.text}")
print(f"Tokens: {result.tokens_generated}")
print(f"Latency: {result.latency_ms:.1f} ms")
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
```

### Generate (Alias)

`generate()` is an alias for `infer()` with identical parameters:

```python
result = engine.generate("What is flash attention?", max_tokens=64)
```

### Batch Inference

Process multiple prompts in a single call:

```python
prompts = [
    "What is CUDA?",
    "Explain tensor cores.",
    "What is NCCL?",
]

results = engine.batch_infer(prompts, max_tokens=64, temperature=0.5)

for i, result in enumerate(results):
    if result.success:
        print(f"[{i}] {result.text[:80]}...")
    else:
        print(f"[{i}] ERROR: {result.error_message}")
```

### Streaming Inference

Stream tokens as they are generated, with a callback function:

```python
def on_token(token_text):
    print(token_text, end="", flush=True)

result = engine.infer_stream(
    prompt="Write a haiku about GPUs.",
    callback=on_token,
    max_tokens=64,
)
print()  # Newline after streaming
print(f"Total tokens: {result.tokens_generated}")
```

### Inference Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | required | The input text prompt |
| `max_tokens` | `int` | `128` | Maximum tokens to generate |
| `temperature` | `float` | `0.7` | Sampling temperature (0 = greedy) |
| `top_p` | `float` | `0.9` | Nucleus sampling threshold |
| `top_k` | `int` | `40` | Top-K sampling |
| `seed` | `int` | `0` | Random seed (0 = random) |

## InferResult

Every inference call returns an `InferResult` dataclass:

```python
@dataclass
class InferResult:
    success: bool              # Whether the call succeeded
    text: str                  # Generated text
    tokens_generated: int      # Number of tokens produced
    latency_ms: float          # Wall-clock latency in milliseconds
    tokens_per_sec: float      # Throughput (tokens / second)
    error_message: str | None  # Error details if success is False
```

### Handling Results

```python
result = engine.infer("What is quantization?", max_tokens=128)

if result.success:
    print(result.text)
    print(f"Generated {result.tokens_generated} tokens in {result.latency_ms:.0f} ms")
    print(f"Throughput: {result.tokens_per_sec:.1f} tokens/sec")
else:
    print(f"Inference failed: {result.error_message}")
```

### Common Error Patterns

| Error | Cause | Fix |
|-------|-------|-----|
| `success=False`, connection error | Server not running | Set `auto_start=True` in `load_model()` |
| `success=False`, timeout | Model too large or prompt too long | Reduce `max_tokens` or use a smaller model |
| `tokens_generated=0` | Empty generation | Increase temperature or adjust prompt |

## Metrics Collection

The engine maintains in-process performance metrics across all inference calls:

```python
engine.infer("Prompt 1", max_tokens=64)
engine.infer("Prompt 2", max_tokens=64)
engine.infer("Prompt 3", max_tokens=64)

metrics = engine.get_metrics()
print(f"Total requests: {metrics['requests']}")
print(f"Total tokens: {metrics['total_tokens']}")
print(f"Total latency: {metrics['total_latency_ms']:.1f} ms")
print(f"Per-call latencies: {metrics['latencies']}")
```

These metrics are lightweight counters maintained in Python. For production observability, enable OpenTelemetry telemetry (see below).

## Telemetry Integration

Enable OpenTelemetry instrumentation to export traces and metrics to Grafana Cloud, Jaeger, or any OTLP-compatible backend:

```python
import llamatelemetry as lt

engine = lt.InferenceEngine(
    server_url="http://127.0.0.1:8090",
    enable_telemetry=True,
    telemetry_config={
        "service_name": "my-llm-service",
        "service_version": "0.1.1",
        "otlp_endpoint": "http://localhost:4317",
        "enable_llama_metrics": True,
        "llama_metrics_interval": 5.0,
    },
)
```

When telemetry is enabled, every `infer()` call creates an OpenTelemetry span with 45 `gen_ai.*` semantic attributes, including model name, token counts, latency, and GPU utilization.

See the [Telemetry and Observability](telemetry-observability.md) guide for full details.

## Cleanup and Lifecycle

### Manual Cleanup

```python
engine.unload_model()  # Stops the server and releases resources
```

### Context Manager (Recommended)

```python
with lt.InferenceEngine() as engine:
    engine.load_model("gemma-3-1b-Q4_K_M")
    result = engine.infer("Hello!", max_tokens=32)
# Automatic cleanup here
```

### Lifecycle Flow

1. `InferenceEngine()` -- creates the engine, optionally initializes telemetry
2. `load_model()` -- resolves the model, starts the server, waits for readiness
3. `infer()` / `generate()` / `batch_infer()` / `infer_stream()` -- runs inference
4. `unload_model()` or context manager exit -- stops the server, flushes telemetry

## Best Practices

- **Always use the context manager** to avoid orphaned server processes.
- **Use `auto_configure=True`** unless you need precise control over GPU layers and context size.
- **Prefer registry names** over raw paths for reproducibility across environments.
- **Enable telemetry in production** to capture latency distributions and throughput trends.
- **Use `batch_infer()`** for offline evaluation workloads to improve throughput.
- **Set `seed`** to a nonzero value for reproducible outputs during testing.

## Complete Example

```python
import llamatelemetry as lt

with lt.InferenceEngine(
    server_url="http://127.0.0.1:8090",
    enable_telemetry=True,
    telemetry_config={
        "service_name": "demo",
        "otlp_endpoint": "http://localhost:4317",
    },
) as engine:
    # Load a model from the registry
    engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True, auto_configure=True)

    # Single inference
    result = engine.infer("What is GGUF?", max_tokens=128, temperature=0.7)
    print(result.text)

    # Batch inference
    results = engine.batch_infer(
        ["What is CUDA?", "What is NCCL?"],
        max_tokens=64,
    )
    for r in results:
        print(r.text[:100])

    # Check metrics
    metrics = engine.get_metrics()
    print(f"Completed {metrics['requests']} requests")
```

## Related

- [Server Management](server-management.md) -- direct server control
- [Model Management](model-management.md) -- model registry and downloads
- [API Client](api-client.md) -- low-level HTTP client
- [Core API Reference](../reference/core-api.md)
