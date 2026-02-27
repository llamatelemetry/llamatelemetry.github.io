# llamatelemetry Documentation v0.1.0

`llamatelemetry` is a CUDA-first Python SDK for local LLM inference and observability around `llama.cpp`/GGUF, with optional OpenTelemetry, Kaggle workflows, Graphistry analytics, and GPU utilities.

## What this documentation includes

- Practical setup for local and Kaggle environments
- End-to-end inference and server lifecycle guides
- OpenAI-compatible and native llama.cpp API client usage
- GGUF model management and quantization workflows
- Multi-GPU and NCCL guidance
- Notebook learning paths (16 notebooks)
- Detailed API reference across all modules

## Quick links

- [Get Started Overview](get-started/index.md)
- [Installation](get-started/installation.md)
- [Quickstart](get-started/quickstart.md)
- [Kaggle Quickstart](get-started/kaggle-quickstart.md)
- [Troubleshooting](guides/troubleshooting.md)
- [API Reference Index](reference/index.md)
- [Notebook Hub](notebooks/index.md)

## Core package surfaces

- `InferenceEngine` for high-level loading/inference
- `ServerManager` for `llama-server` lifecycle
- `llamatelemetry.api.LlamaCppClient` for endpoint-level control
- `llamatelemetry.telemetry` for traces/metrics/export
- `llamatelemetry.kaggle` for zero-boilerplate Kaggle setup

## Example

```python
from llamatelemetry import InferenceEngine

engine = InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)
result = engine.infer("Explain llama.cpp in 2 sentences.", max_tokens=100)

if result.success:
    print(result.text)
else:
    print(result.error_message)
```

## Documentation assumptions

- This site targets `llamatelemetry` version `0.1.0`.
- Features can be optional based on environment (`GPU`, `OpenTelemetry`, `Graphistry`, `Triton`, `Unsloth`).
- For notebook-focused workflows, Kaggle dual T4 is the primary optimized path.
