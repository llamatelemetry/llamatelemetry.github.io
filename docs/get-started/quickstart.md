---
title: llamatelemetry Quickstart
description: Minimal quickstart for loading a GGUF model with llamatelemetry, starting llama-server automatically, running inference, and inspecting basic metrics.
---

# Quickstart

This quickstart keeps to the **most reliable path in the current SDK**:

1. import the package
2. verify CUDA visibility
3. create an `InferenceEngine`
4. load a GGUF model
5. run inference
6. inspect basic metrics

For Kaggle-specific setup, use the separate [Kaggle Quickstart](kaggle-quickstart.md).

## 1. Import and inspect the environment

```python
import llamatelemetry as lt

print("version:", lt.__version__)
print("server path:", lt.get_llama_cpp_cuda_path())
print("cuda info:", lt.detect_cuda())
```

The important thing here is not matching some exact printed format. The goal is
simply to confirm that the package imports, sees your runtime, and can find the
`llama-server` path it expects to use.

## 2. Create an inference engine

```python
import llamatelemetry as lt

engine = lt.InferenceEngine(
    server_url="http://127.0.0.1:8080",
    enable_telemetry=False,
)
```

`InferenceEngine` is the highest-level API in the current package.

## 3. Load a model

The SDK supports three practical model-loading patterns.

### Option A: built-in registry name

```python
engine.load_model(
    "gemma-3-1b-Q4_K_M",
    auto_start=True,
    auto_configure=True,
    verbose=True,
)
```

### Option B: Hugging Face repo plus file

```python
engine.load_model(
    "bartowski/gemma-2-2b-it-GGUF:gemma-2-2b-it-Q4_K_M.gguf",
    auto_start=True,
)
```

### Option C: local GGUF file

```python
engine.load_model(
    "/path/to/model.gguf",
    auto_start=True,
    gpu_layers=99,
    ctx_size=4096,
)
```

The safest documentation stance is this: use small or moderate GGUF models
first, confirm your runtime is stable, then move to larger models.

## 4. Run one inference request

```python
result = engine.infer(
    prompt="Explain what GGUF is in two sentences.",
    max_tokens=128,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
)

print("success:", result.success)
print("text:", result.text)
print("tokens:", result.tokens_generated)
print("latency_ms:", result.latency_ms)
print("tokens_per_sec:", result.tokens_per_sec)
```

The `InferResult` object is one of the strongest parts of the current public API
because it gives you the generated text plus simple performance signals in one
place.

## 5. Run batch inference

The current SDK exposes `batch_infer()`.

```python
prompts = [
    "Define CUDA in one sentence.",
    "Define quantization in one sentence.",
    "Define observability in one sentence.",
]

results = engine.batch_infer(prompts, max_tokens=64)

for i, r in enumerate(results):
    print("---", i)
    print("success:", r.success)
    print("text:", r.text)
```

## 6. Inspect engine metrics

```python
metrics = engine.get_metrics()
print(metrics)
```

This is the simplest way to see the in-process aggregate counters the engine has
collected during your session.

## 7. Check server-side endpoints when needed

If the underlying `llama-server` is running, `ServerManager` and the lower-level
client APIs can expose health and metrics endpoints. A simple pattern is:

```python
from llamatelemetry import ServerManager

manager = ServerManager(server_url="http://127.0.0.1:8080")
print(manager.check_server_health())
print(manager.get_health())
```

You can also query the Prometheus-style metrics text:

```python
print(manager.get_metrics())
```

## 8. Clean up

```python
engine.unload_model()
```

Or use a context manager:

```python
import llamatelemetry as lt

with lt.InferenceEngine(enable_telemetry=False) as engine:
    engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)
    result = engine.infer("Hello from llamatelemetry.")
    print(result.text)
```

## A realistic first workflow

For a first successful run, this sequence is usually enough:

```python
import llamatelemetry as lt

engine = lt.InferenceEngine(enable_telemetry=False)
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)
result = engine.infer("What does this SDK do?", max_tokens=96)
print(result.text)
print(engine.get_metrics())
engine.unload_model()
```

## What this page intentionally does not claim

This page avoids a few claims that were too broad in the earlier docs:

- it does not claim token streaming as a stable top-level `InferenceEngine` API
  because the uploaded snapshot does not expose `stream_infer()` on that class
- it does not claim every advanced integration is equally validated
- it does not assume all local machines behave like Kaggle dual-T4 notebooks

For telemetry-specific setup, continue to the
[Telemetry and Observability Guide](../guides/telemetry-observability.md).
