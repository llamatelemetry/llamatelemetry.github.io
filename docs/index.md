# llamatelemetry Documentation

**CUDA-first Python SDK for local LLM inference and observability with llama.cpp**

---

`llamatelemetry` is a comprehensive Python SDK (v0.1.0) that wraps `llama.cpp` with a high-level inference API, robust server lifecycle management, OpenTelemetry instrumentation, Kaggle-optimized presets, and GPU-accelerated graph analytics. It is designed for researchers and engineers who need production-grade LLM inference observability on NVIDIA GPUs.

## Key Features

| Feature | Description |
|---------|-------------|
| **InferenceEngine** | High-level API: load models, start servers, run inference in 3 lines |
| **ServerManager** | Full `llama-server` lifecycle: start, stop, health, metrics, slots |
| **LlamaCppClient** | OpenAI-compatible chat completions, embeddings, reranking, tokenization |
| **Multi-GPU** | Layer-split and row-split inference across multiple GPUs with NCCL |
| **OpenTelemetry** | 45 `gen_ai.*` span attributes, 5 metrics, OTLP export to Grafana/Jaeger |
| **Kaggle Presets** | Zero-config dual-T4 setup with `ServerPreset.KAGGLE_DUAL_T4` |
| **GGUF Tooling** | Model reports, suitability checks, quantization matrix, size estimates |
| **Graphistry** | GPU-accelerated graph visualization of traces, embeddings, and knowledge graphs |
| **Model Registry** | 22+ curated GGUF models with auto-download from HuggingFace |
| **Quantization** | NF4, dynamic quantization, GGUF conversion from PyTorch |

## Quickstart

```python
from llamatelemetry import InferenceEngine

engine = InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)

result = engine.infer("Explain llama.cpp in two sentences.", max_tokens=96)
print(result.text)
print(f"{result.tokens_per_sec:.1f} tok/s | {result.latency_ms:.0f} ms")

engine.unload_model()
```

## Quickstart with Telemetry

```python
from llamatelemetry import InferenceEngine

engine = InferenceEngine(
    enable_telemetry=True,
    telemetry_config={
        "service_name": "my-llm-service",
        "otlp_endpoint": "http://localhost:4317",
        "enable_llama_metrics": True,
    },
)
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)
result = engine.generate("What is CUDA?", max_tokens=64)
# Spans and metrics are automatically exported to your OTLP backend
```

## Kaggle Quickstart

```python
from llamatelemetry.kaggle.pipeline import (
    start_server_from_preset,
    setup_otel_and_client,
    load_grafana_otlp_env_from_kaggle,
    KagglePipelineConfig,
)
from llamatelemetry.kaggle import ServerPreset

load_grafana_otlp_env_from_kaggle()
manager = start_server_from_preset("/kaggle/input/model/model.gguf", ServerPreset.KAGGLE_DUAL_T4)

cfg = KagglePipelineConfig(enable_llama_metrics=True)
ctx = setup_otel_and_client("http://127.0.0.1:8080", cfg)
client = ctx["client"]

resp = client.chat_completions({
    "model": "local-gguf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64,
})
print(resp.choices[0].message.content)
```

## Documentation Sections

### [Get Started](get-started/index.md)
Installation, environment setup, and your first inference in under 5 minutes.

### [Guides](guides/inference-engine.md)
In-depth tutorials covering every SDK module: inference, server management, multi-GPU, telemetry, Graphistry, quantization, and more.

### [API Reference](reference/index.md)
Complete API documentation with full signatures, parameter descriptions, return types, and code examples for every public class and function.

### [Notebooks](notebooks/index.md)
18 Kaggle-ready Jupyter notebooks organized into four learning tracks: Foundation, Integration, Advanced, and Observability.

### [Project](project/architecture.md)
Architecture overview, file map, release artifacts, FAQ, changelog, and contributing guidelines.

## Architecture Overview

```
llamatelemetry (Python SDK)
    |
    |-- InferenceEngine          # High-level facade
    |-- ServerManager            # llama-server process lifecycle
    |-- LlamaCppClient           # OpenAI-compatible HTTP client
    |-- MultiGPUConfig + NCCL    # Multi-GPU orchestration
    |-- Telemetry                # OpenTelemetry spans + metrics
    |-- Kaggle Pipeline          # Presets + secrets + pipeline helpers
    |-- Graphistry               # GPU-accelerated graph analytics
    |-- Quantization             # NF4, GGUF conversion, dynamic quant
    |-- _internal/bootstrap      # Auto-download binaries (~961 MB)
    |
    v
llama-server (C++ binary)       # llama.cpp HTTP server
    |
    v
CUDA / cuBLAS / NCCL            # GPU compute layer
```

## Requirements

- **Python** >= 3.11
- **CUDA** 12.x with NVIDIA GPU (compute capability >= 7.5)
- **Target GPU:** Tesla T4 (SM 7.5) — optimized for Kaggle dual-T4
- **OS:** Linux (tested on Ubuntu 22.04+)

## License

MIT License. Copyright 2024 Waqas Muhammad.

## Links

- **GitHub:** [llamatelemetry/llamatelemetry](https://github.com/llamatelemetry/llamatelemetry)
- **PyPI:** `pip install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0`
- **Notebooks:** [18 Kaggle notebooks](notebooks/index.md)
