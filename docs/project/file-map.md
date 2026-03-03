# Project File Map

This page provides a complete annotated map of every Python file in the
`llamatelemetry` package, grouped by module. Each entry includes a one-line
description of the file's purpose.

---

## Top-Level Layout

```
llamatelemetry/              Main repository root
  llamatelemetry/            Python package (source)
  csrc/                      C++/CUDA extension sources
  core/                      Tensor and device primitives (build output)
  docs/                      MkDocs content (in-repo documentation)
  examples/                  Runnable example scripts
  notebooks/                 18 curated Kaggle-ready notebooks
  tests/                     Unit and end-to-end tests (246 passing, 24 skipped)
  scripts/                   Release and HuggingFace helper scripts
  releases/                  Source and binary release artifacts
  pyproject.toml             Build configuration (setuptools + CMake)
  README.md                  Project README
  LICENSE                    MIT License
```

---

## Python Package (`llamatelemetry/`)

### Root Files

| File | Description |
|------|-------------|
| `__init__.py` | Package entry point: `InferenceEngine`, `InferResult`, bootstrap integration, auto-configuration of `LD_LIBRARY_PATH` and `LLAMA_SERVER_PATH` (~1014 lines) |
| `server.py` | `ServerManager` class: llama-server lifecycle management (start, stop, health check, binary download) (~910 lines) |
| `models.py` | Model registry helpers, metadata loading, and model resolution utilities |
| `gguf_parser.py` | Standalone GGUF file format parser for metadata extraction |
| `chat.py` | Chat-specific helper functions and conversation formatting |
| `embeddings.py` | Embedding generation helpers and vector utilities |
| `jupyter.py` | Jupyter/IPython display helpers and notebook integration utilities |
| `utils.py` | Shared utility functions: `detect_cuda()`, `check_gpu_compatibility()`, `find_gguf_models()`, `print_system_info()`, `load_config()`, `get_recommended_gpu_layers()` |

### `api/` -- Server API Client (4 files)

| File | Description |
|------|-------------|
| `api/__init__.py` | Module exports: `LlamaCppClient`, `MultiGPUConfig`, GGUF utilities, NCCL bindings (with graceful stubs when NCCL is unavailable) |
| `api/client.py` | `LlamaCppClient` class: unified OpenAI-compatible + native llama.cpp API client with chat, completions, embeddings, reranking, tokenization, LoRA, slot, and health endpoints (~1293 lines) |
| `api/multigpu.py` | `MultiGPUConfig`, `SplitMode`, `GPUInfo`: GPU detection, VRAM estimation, preset configurations for Kaggle T4 x2 and Colab T4 (~663 lines) |
| `api/gguf.py` | GGUF model utilities: `parse_gguf_header()`, `quantize()`, `convert_hf_to_gguf()`, `merge_lora()`, `gguf_report()`, `report_model_suitability()`, quantization matrix |
| `api/nccl.py` | `NCCLCommunicator`: NCCL collective operations (AllReduce, Broadcast, AllGather, ReduceScatter) via ctypes bindings to `libnccl.so` (~645 lines) |

### `telemetry/` -- OpenTelemetry Integration (10 files)

| File | Description |
|------|-------------|
| `telemetry/__init__.py` | Module entry: `setup_telemetry()`, `setup_grafana_otlp()`, lazy OTel/Graphistry imports, `InstrumentedLLMClient` convenience wrapper |
| `telemetry/semconv.py` | OpenTelemetry GenAI semantic convention helpers: 45 `gen_ai.*` attribute constants with safe fallbacks |
| `telemetry/resource.py` | `build_gpu_resource()`: constructs OTel Resource with GPU info, service name, llama-server metadata |
| `telemetry/tracer.py` | `InferenceTracerProvider`: wraps OTel `TracerProvider` with GPU-aware resource detection and optional Graphistry export |
| `telemetry/metrics.py` | `GpuMetricsCollector`: exports GPU metrics (latency histograms, tokens/sec gauges, VRAM usage, NCCL stats) via OTel MeterProvider |
| `telemetry/exporter.py` | `build_exporters()`: configures OTLP span exporters (gRPC/HTTP) for vendor-neutral telemetry export |
| `telemetry/auto_instrument.py` | Decorators and context managers: `instrument_inference()`, `inference_span()`, `batch_inference_span()`, `create_llm_attributes()` |
| `telemetry/instrumentor.py` | `LlamaCppClientInstrumentor`: monkey-patch instrumentor for `LlamaCppClient` methods, auto-emitting spans and metrics |
| `telemetry/client.py` | `InstrumentedLlamaCppClient`, `InstrumentationConfig`: pre-instrumented client subclass with configurable telemetry |
| `telemetry/graphistry_export.py` | `GraphistryTraceExporter`: real-time trace span export to Graphistry for graph-based trace visualization |
| `telemetry/monitor.py` | `PerformanceMonitor`, `PerformanceSnapshot`, `InferenceRecord`: real-time inference performance tracking and summarization |

### `kaggle/` -- Kaggle Environment Utilities (5 files)

| File | Description |
|------|-------------|
| `kaggle/__init__.py` | Module exports: `KaggleEnvironment`, presets, secrets, GPU context, pipeline helpers |
| `kaggle/environment.py` | `KaggleEnvironment`: one-liner setup for Kaggle notebooks, `quick_setup()` alias, engine creation with optimal settings |
| `kaggle/presets.py` | `ServerPreset`, `TensorSplitMode`, `PresetConfig`: pre-configured server settings for common model sizes on T4 x2 |
| `kaggle/secrets.py` | `KaggleSecrets`: auto-load secrets from Kaggle, `setup_huggingface_auth()`, `setup_graphistry_auth()` |
| `kaggle/gpu_context.py` | `GPUContext`, `split_gpu_session()`, `rapids_gpu()`, `llm_gpu()`: GPU isolation context managers for split-GPU workflows |
| `kaggle/pipeline.py` | `KagglePipelineConfig`, `load_grafana_otlp_env_from_kaggle()`, `start_server_from_preset()`, `setup_otel_and_client()`: end-to-end pipeline helpers |

### `inference/` -- Inference Optimization (4 files)

| File | Description |
|------|-------------|
| `inference/__init__.py` | Module exports for inference optimization classes |
| `inference/flash_attn.py` | FlashAttention configuration and helpers for memory-efficient attention |
| `inference/kv_cache.py` | KV cache management: size estimation, eviction policies, memory budgeting |
| `inference/batch.py` | `ContinuousBatching`: batch scheduling and slot management for concurrent requests |

### `cuda/` -- CUDA Utilities (4 files)

| File | Description |
|------|-------------|
| `cuda/__init__.py` | Module exports for CUDA utility classes |
| `cuda/graphs.py` | `CUDAGraph`: CUDA graph capture and replay for reduced kernel launch overhead |
| `cuda/triton_kernels.py` | Triton kernel wrappers for custom GPU compute operations |
| `cuda/tensor_core.py` | TensorCore utilities for mixed-precision matrix operations on SM 7.5+ |

### `quantization/` -- Quantization Helpers (4 files)

| File | Description |
|------|-------------|
| `quantization/__init__.py` | Module exports for quantization utilities |
| `quantization/nf4.py` | NF4 (4-bit NormalFloat) quantization implementation |
| `quantization/gguf.py` | GGUF quantization format conversion and manipulation |
| `quantization/dynamic.py` | Dynamic quantization: runtime precision selection based on available VRAM |

### `graphistry/` -- Graph Visualization (5 files)

| File | Description |
|------|-------------|
| `graphistry/__init__.py` | Module exports: `GraphWorkload`, `RAPIDSBackend`, `GraphistryConnector`, visualization classes |
| `graphistry/workload.py` | `GraphWorkload`, `SplitGPUManager`: GPU-partitioned graph workload management, knowledge graph creation |
| `graphistry/connector.py` | `GraphistryConnector`, `GraphistrySession`: authentication and session management for PyGraphistry |
| `graphistry/rapids.py` | `RAPIDSBackend`: cuDF DataFrame creation, cuGraph algorithm execution, RAPIDS availability detection |
| `graphistry/viz.py` | `GraphistryViz`, `TraceVisualization`, `MetricsVisualization`: high-level graph visualization builders |
| `graphistry/builders.py` | `GraphistryBuilders`: DataFrame builders for inference records, traces, node/edge generation, latency time series |

### `louie/` -- AI Graph Analysis (3 files)

| File | Description |
|------|-------------|
| `louie/__init__.py` | Module exports for Louie.AI integration |
| `louie/client.py` | Louie.AI API client for AI-powered graph query and analysis |
| `louie/knowledge.py` | Knowledge extraction: entity and relationship extraction from LLM output for graph construction |

### `unsloth/` -- Fine-Tuning Integration (4 files)

| File | Description |
|------|-------------|
| `unsloth/__init__.py` | Module exports for Unsloth integration |
| `unsloth/loader.py` | `UnslothModelLoader`: load models for fine-tuning with Unsloth |
| `unsloth/adapter.py` | LoRA adapter management: apply, merge, and configure adapters |
| `unsloth/exporter.py` | `UnslothExporter`, `ExportConfig`: export fine-tuned models to GGUF format for llama.cpp |

### `_internal/` -- Bootstrap and Registry (3 files)

| File | Description |
|------|-------------|
| `_internal/__init__.py` | Internal module marker |
| `_internal/bootstrap.py` | `bootstrap()`: auto-download CUDA binaries (~961 MB) from GitHub Releases on first import |
| `_internal/registry.py` | `MODEL_REGISTRY`: curated dictionary of 30+ GGUF models with repo, filename, size, VRAM requirements, and descriptions |

---

## C++/CUDA Extension (`csrc/`)

| File | Description |
|------|-------------|
| `csrc/bindings.cpp` | pybind11 module definition for `llamatelemetry_cpp` |
| `csrc/core/device.h` | CUDA device discovery and management (header) |
| `csrc/core/device.cu` | CUDA device discovery and management (implementation) |
| `csrc/core/tensor.h` | Tensor RAII class with 6 DTypes: float32, float16, bfloat16, int32, int8, uint8 (header) |
| `csrc/core/tensor.cu` | Tensor RAII class (CUDA implementation) |
| `csrc/ops/matmul.h` | cuBLAS matrix multiplication interface: SGEMM (float32) and HGEMM (float16) (header) |
| `csrc/ops/matmul.cu` | cuBLAS matrix multiplication kernels (CUDA implementation) |

---

## File Count Summary

| Category | Files | Approximate Lines |
|----------|-------|-------------------|
| Root Python files | 8 | ~3,000 |
| `api/` | 5 | ~2,800 |
| `telemetry/` | 11 | ~2,500 |
| `kaggle/` | 6 | ~1,200 |
| `inference/` | 4 | ~800 |
| `cuda/` | 4 | ~600 |
| `quantization/` | 4 | ~500 |
| `graphistry/` | 6 | ~1,000 |
| `louie/` | 3 | ~400 |
| `unsloth/` | 4 | ~500 |
| `_internal/` | 3 | ~700 |
| C++/CUDA (`csrc/`) | 7 | ~650 |
| **Total** | **~65** | **~14,650** |

---

## Where to Go Next

- [Architecture Overview](architecture.md) -- module interactions and data flow
- [Release Artifacts](release-artifacts.md) -- what ships in each release bundle
- [Core API Reference](../reference/core-api.md) -- detailed class and function documentation
