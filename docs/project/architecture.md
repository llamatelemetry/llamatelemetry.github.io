---
title: llamatelemetry Architecture Overview
description: Architecture overview of llamatelemetry, including the Python SDK modules, llama-server integration, C++ and CUDA extension layer, and observability pipeline.
---

# Architecture Overview

`llamatelemetry` is a CUDA-first Python SDK that orchestrates `llama-server` for GGUF
model inference with optional OpenTelemetry observability, Kaggle environment management,
and GPU-accelerated graph analytics. The SDK is built on approximately 40 Python files
(13,000+ lines) across 10 modules, backed by 7 C++/CUDA source files (~650 lines).

---

## High-Level Architecture

```
+-----------------------------------------------------------------------+
|                        User Application                               |
|  engine = InferenceEngine()                                           |
|  engine.load_model("gemma-3-1b-Q4_K_M")                              |
|  result = engine.infer("What is AI?", max_tokens=100)                 |
+-----------------------------------------------------------------------+
        |                    |                        |
        v                    v                        v
+----------------+  +------------------+  +------------------------+
| InferenceEngine|  | Telemetry Layer  |  | Kaggle Environment     |
| (__init__.py)  |  | (telemetry/)     |  | (kaggle/)              |
| - load_model() |  | - setup_telemetry|  | - KaggleEnvironment    |
| - infer()      |  | - OTel spans     |  | - split_gpu_session    |
| - InferResult  |  | - GPU metrics    |  | - ServerPreset         |
+-------+--------+  +--------+---------+  +-----------+------------+
        |                     |                        |
        v                     v                        v
+----------------+  +------------------+  +------------------------+
| ServerManager  |  | LlamaCppClient   |  | Graphistry / RAPIDS    |
| (server.py)    |  | (api/client.py)  |  | (graphistry/)          |
| - start/stop   |  | - chat API       |  | - GraphWorkload        |
| - health check |  | - completions    |  | - RAPIDSBackend        |
| - auto-download|  | - embeddings     |  | - knowledge graphs     |
+-------+--------+  +--------+---------+  +-----------+------------+
        |                     |                        |
        v                     v                        v
+-----------------------------------------------------------------------+
|                       llama-server (binary)                           |
|  - OpenAI-compatible REST API                                         |
|  - Multi-slot continuous batching                                     |
|  - CUDA 12 GPU acceleration (SM 7.5 / Tesla T4)                      |
+-----------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------+
|                    C++/CUDA Extension (csrc/)                         |
|  - llamatelemetry_cpp pybind11 module                                 |
|  - Tensor RAII (6 DTypes), Device ops                                 |
|  - cuBLAS matmul (SGEMM / HGEMM)                                     |
+-----------------------------------------------------------------------+
```

---

## 10-Module Structure

The Python package is organized into 10 self-contained modules plus several
top-level utility files.

### Core Modules

| Module | Files | Purpose |
|--------|-------|---------|
| `__init__.py` + `server.py` | 2 | `InferenceEngine`, `InferResult`, `ServerManager` -- high-level API and server lifecycle |
| `api/` | 4 | `LlamaCppClient` (OpenAI-compatible + native), `MultiGPUConfig`, GGUF utilities, NCCL bindings |
| `telemetry/` | 10 | OpenTelemetry integration: 45 `gen_ai.*` attributes, 5 metrics, auto-instrumentation, Graphistry export |
| `kaggle/` | 5 | `KaggleEnvironment`, `split_gpu_session`, server presets, secrets management, GPU context isolation |

### Inference and GPU Modules

| Module | Files | Purpose |
|--------|-------|---------|
| `inference/` | 4 | FlashAttention helpers, KV cache management, continuous batching |
| `cuda/` | 4 | CUDA graph capture, Triton kernel wrappers, TensorCore utilities |
| `quantization/` | 4 | NF4 quantization, GGUF conversion, dynamic quantization |

### Analytics and Integration Modules

| Module | Files | Purpose |
|--------|-------|---------|
| `graphistry/` | 5 | GPU-accelerated graph visualization, RAPIDS cuGraph, split-GPU workload management |
| `louie/` | 3 | AI-powered graph analysis, knowledge extraction via Louie.AI |
| `unsloth/` | 4 | Fine-tuning integration, LoRA adapter management, GGUF export |

### Internal Module

| Module | Files | Purpose |
|--------|-------|---------|
| `_internal/` | 3 | Bootstrap (auto-download ~961 MB of CUDA binaries), `MODEL_REGISTRY` with 30+ curated GGUF models |

---

## C++/CUDA Layer (`csrc/`)

The native extension is built as `llamatelemetry_cpp` via pybind11 and CMake.

```
csrc/
  bindings.cpp          pybind11 module definition
  core/
    device.h / device.cu   CUDA device discovery and management
    tensor.h / tensor.cu   Tensor RAII class (6 DTypes: float32, float16, bfloat16, int32, int8, uint8)
  ops/
    matmul.h / matmul.cu   cuBLAS matrix multiplication (SGEMM for float32, HGEMM for float16)
```

**Link dependencies:** `cudart_static`, `cublas_static`, `cublasLt_static`

The C++/CUDA layer is compiled during the Kaggle build pipeline and distributed
as part of the CUDA binary tar.gz artifact. It is not required for basic
Python-only usage of the SDK.

---

## Data Flow: Inference Request

The following sequence shows what happens when `engine.infer()` is called.

```
User Code
  |
  |  engine.infer("What is AI?", max_tokens=100)
  v
InferenceEngine.__init__.py
  |
  |  1. Resolve model (registry lookup or local path)
  |  2. Ensure ServerManager is running
  |  3. Build request payload
  v
LlamaCppClient (api/client.py)
  |
  |  4. POST /completion or /v1/chat/completions
  |  5. Parse JSON response into dataclasses
  v
ServerManager (server.py)
  |
  |  6. Forward to llama-server process
  |  7. llama-server runs CUDA inference
  v
InferResult
  |
  |  8. Wrap response: success, text, tokens_generated,
  |     latency_ms, tokens_per_sec
  v
(Optional) Telemetry
  |
  |  9. Emit gen_ai.* span attributes
  | 10. Record metrics (latency, throughput, VRAM)
  v
Return to User
```

---

## Data Flow: Bootstrap Sequence

On first import, the SDK auto-configures itself.

```
import llamatelemetry
  |
  |  1. Set LD_LIBRARY_PATH to include bundled libs
  |  2. Set LLAMA_SERVER_PATH to bundled binary
  v
Binary exists?
  |
  +-- YES --> Ready
  |
  +-- NO  --> _internal.bootstrap.bootstrap()
                |
                |  3. Download CUDA bundle from GitHub Releases
                |     (~961 MB: llama-server, llama-cli, libnccl.so, etc.)
                |  4. Extract to llamatelemetry/binaries/cuda12/
                |  5. Set environment variables
                v
              Ready
```

---

## Design Patterns

### Context Managers

The SDK uses Python context managers throughout for resource lifecycle management.

```python
# InferenceEngine
with InferenceEngine() as engine:
    engine.load_model("gemma-3-1b-Q4_K_M")
    result = engine.infer("Hello")

# ServerManager
with ServerManager() as server:
    server.start_server(model_path="model.gguf")

# NCCLCommunicator
with NCCLCommunicator(config) as comm:
    comm.all_reduce(tensor)

# GPU context isolation
with split_gpu_session(llm_gpu=0, rapids_gpu=1):
    # GPU 0 for inference, GPU 1 for analytics
    pass
```

### Lazy Loading and Optional Dependencies

Modules that depend on optional packages use try/except imports with graceful
fallbacks. This keeps the core SDK lightweight.

```python
# telemetry/__init__.py
_OTEL_AVAILABLE = False
try:
    from opentelemetry import trace, metrics
    _OTEL_AVAILABLE = True
except ImportError:
    pass

# graphistry/__init__.py -- similar pattern for pygraphistry
# api/nccl.py -- similar pattern for NCCL shared library
```

### Graceful Degradation

When an optional dependency is missing, the SDK warns the user but continues
to function. For example, `setup_telemetry()` returns `(None, None)` if
OpenTelemetry is not installed, and NCCL functions fall back to stubs when
`libnccl.so` is unavailable.

### Dataclass-Driven API

The SDK uses 15+ dataclasses and enums for structured data:

- `InferResult` -- inference response wrapper
- `Message`, `Choice`, `Usage`, `Timings` -- chat completion types
- `CompletionResponse`, `EmbeddingsResponse` -- API response types
- `MultiGPUConfig`, `GPUInfo`, `SplitMode` -- GPU configuration
- `NCCLConfig`, `NCCLInfo` -- NCCL configuration
- `GGUFMetadata`, `GGUFTensorInfo`, `GGMLType` -- model metadata
- `ServerPreset`, `PresetConfig` -- server configuration presets
- `PerformanceSnapshot`, `InferenceRecord` -- monitoring types

### PyTorch-Style Auto-Configuration

On import, the package automatically:

1. Detects and configures `LD_LIBRARY_PATH` for bundled shared libraries
2. Locates or downloads the `llama-server` binary
3. Sets `LLAMA_SERVER_PATH` in the environment
4. Creates model cache directories

This means users can start with `import llamatelemetry` and immediately begin
inference without manual path configuration.

---

## Dependency Graph

```
llamatelemetry (core)
  +-- numpy
  +-- requests
  +-- huggingface_hub
  +-- tqdm
  +-- opentelemetry-api (optional)
  +-- opentelemetry-sdk (optional)
  |
  +-- telemetry/ --> opentelemetry-exporter-otlp (optional)
  +-- graphistry/ --> pygraphistry, pandas (optional)
  +-- graphistry/ --> cudf, cugraph (optional, RAPIDS)
  +-- cuda/ --> triton (optional)
  +-- unsloth/ --> torch, unsloth (optional)
  +-- api/nccl.py --> libnccl.so (optional, via ctypes)
  +-- api/client.py --> sseclient (optional, for streaming)
  +-- kaggle/ --> ipywidgets (optional)
  +-- telemetry/monitor.py --> pynvml (optional, GPU metrics)
```

---

## Runtime Characteristics

- **Kaggle-first**: optimized for Tesla T4 x2 dual-GPU environments (SM 7.5, 30 GB total VRAM)
- **Hybrid bootstrap**: CUDA binaries are downloaded on first use from GitHub Releases
- **Split-GPU architecture**: GPU 0 runs LLM inference, GPU 1 runs Graphistry/RAPIDS analytics
- **Small model focus**: optimized for 1B--5B parameter GGUF models
- **Python 3.11+**: uses modern Python features and type hints throughout

---

## Where to Go Next

- [Project File Map](file-map.md) -- every file in the package annotated
- [Release Artifacts](release-artifacts.md) -- source and binary release contents
- [Core API Reference](../reference/core-api.md) -- InferenceEngine, ServerManager, LlamaCppClient
- [Telemetry Guide](../guides/telemetry-observability.md) -- OpenTelemetry setup and usage
