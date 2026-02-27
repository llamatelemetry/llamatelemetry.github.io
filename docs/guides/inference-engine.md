# Inference Engine Guide

`InferenceEngine` is the highest-level API in `llamatelemetry`.

## Constructor

```python
from llamatelemetry import InferenceEngine

engine = InferenceEngine(
    server_url="http://127.0.0.1:8090",
    enable_telemetry=False,
    telemetry_config=None,
)
```

## Model loading

`load_model(...)` supports auto-start and auto-configuration:

```python
engine.load_model(
    "gemma-3-1b-Q4_K_M",
    auto_start=True,
    auto_configure=True,
    interactive_download=True,
    n_parallel=1,
)
```

You can pass additional server args via `**kwargs`, for example:

- `batch_size`
- `ubatch_size`
- `flash_attn`
- `tensor_split`

## Inference patterns

Single prompt:

```python
result = engine.infer("Explain GGUF in plain English.")
```

Streaming style callback:

```python
def on_chunk(text):
    print(text)

result = engine.infer_stream("Write a short poem.", callback=on_chunk)
```

Batch prompts:

```python
prompts = ["What is CUDA?", "What is NCCL?", "What is KV cache?"]
results = engine.batch_infer(prompts, max_tokens=80)
```

## Result object

`InferResult` contains:

- `success`
- `text`
- `tokens_generated`
- `latency_ms`
- `tokens_per_sec`
- `error_message`

## Metrics API

```python
metrics = engine.get_metrics()
engine.reset_metrics()
```

Latency includes `mean`, `p50`, `p95`, `p99`.

## Lifecycle

- `engine.is_loaded` tells whether a model is loaded.
- `engine.unload_model()` stops managed server and clears state.
- Context-manager mode auto-cleans on exit.

## Recommended usage pattern

1. Use one engine instance per active server workflow.
2. Use explicit model load/unload around experiments.
3. Use `get_metrics()` after fixed workloads for apples-to-apples benchmarking.
