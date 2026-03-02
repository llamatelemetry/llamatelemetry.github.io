# llamatelemetry Documentation v0.1.0

`llamatelemetry` is a CUDA-first Python SDK for local LLM inference and observability around `llama.cpp`/GGUF. It ships a high-level inference API, a robust `llama-server` lifecycle manager, optional OpenTelemetry instrumentation, Kaggle-first presets, Graphistry/RAPIDS analytics hooks, and GPU optimization utilities.

## What this documentation includes

- Installation and environment setup for local and Kaggle workflows
- Core inference usage (`InferenceEngine`, `ServerManager`, and API client)
- GGUF model management, quantization, and registry workflows
- Multi-GPU, NCCL, and CUDA optimizations
- OpenTelemetry tracing and metrics integration
- Graphistry and knowledge-graph tooling
- 18 notebooks with cell-by-cell walkthroughs
- API reference for all modules and utilities

## Quick links

- [Get Started Overview](get-started/index.md)
- [Installation](get-started/installation.md)
- [Quickstart](get-started/quickstart.md)
- [Kaggle Quickstart](get-started/kaggle-quickstart.md)
- [Troubleshooting](guides/troubleshooting.md)
- [API Reference Index](reference/index.md)
- [Notebook Hub](notebooks/index.md)

## Core package surfaces

- `InferenceEngine` for high-level loading, server bootstrap, and inference
- `ServerManager` for process discovery and `llama-server` lifecycle control
- `llamatelemetry.api.LlamaCppClient` for endpoint-level control and OpenAI-like APIs
- `llamatelemetry.models` for registry-based downloads and model metadata
- `llamatelemetry.telemetry` for traces, metrics, OTLP export, and GPU monitoring
- `llamatelemetry.kaggle` for zero-boilerplate Kaggle setup and presets

## Example

```python
from llamatelemetry import InferenceEngine

engine = InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)
result = engine.infer("Explain llama.cpp in two sentences.", max_tokens=96)

if result.success:
    print(result.text)
else:
    print(result.error_message)
```

## Documentation assumptions

- This site targets `llamatelemetry` version `0.1.0`.
- Features can be optional based on environment (`GPU`, `OpenTelemetry`, `Graphistry`, `Triton`, `Unsloth`).
- For notebook-focused workflows, Kaggle dual T4 is the primary optimized path.
- Release artifacts are documented in the Project section with file maps and binaries.
