# Changelog

All notable changes to `llamatelemetry` are documented in this file.

For the authoritative source-level changelog see the GitHub repository:
[github.com/llamatelemetry/llamatelemetry/blob/main/CHANGELOG.md](https://github.com/llamatelemetry/llamatelemetry/blob/main/CHANGELOG.md)

---

## v0.1.0 — 2026-02-02

Initial public release of the `llamatelemetry` Python SDK. This release establishes the full foundation: a CUDA-first OpenTelemetry SDK for LLM inference observability using llama.cpp as the backend.

### Highlights

- High-level `InferenceEngine` for zero-boilerplate GGUF inference on CUDA GPUs
- Auto-download T4-optimized llama.cpp binary bundle (~961 MB) on first import
- OpenTelemetry tracing and GPU metrics with 45 `gen_ai.*` semantic convention attributes
- Kaggle T4 x2 presets for split-GPU inference workflows
- Graphistry + RAPIDS cuGraph visualization integration
- 18 Jupyter tutorial notebooks spanning foundation to observability workflows
- 246 passing tests across 7 test files
- MIT License

---

### Core Package (`llamatelemetry/__init__.py`)

**Added:**

- `InferenceEngine` — high-level interface for LLM inference lifecycle
  - `load_model(model_name_or_path, ...)` — load GGUF models from registry, local path, or HuggingFace
  - `infer(prompt, max_tokens, temperature, ...)` → `InferResult` — single-shot inference
  - `generate(prompt, ...)` — alias for `infer()`
  - `infer_stream(prompt, ...)` — streaming SSE inference with token callback
  - `batch_infer(prompts, ...)` — parallel batch processing
  - `get_metrics()` / `reset_metrics()` — wall-clock performance tracking
  - `unload_model()` — explicit model cleanup
  - Full context manager support (`with InferenceEngine() as engine:`)
  - Optional OpenTelemetry integration via `enable_telemetry=True`
- `InferResult` — result dataclass with `.success`, `.text`, `.tokens_generated`, `.latency_ms`, `.tokens_per_sec`, `.error_message`
- `is_cuda_available()` — check CUDA availability at runtime
- `get_cuda_device_count()` — count available CUDA devices

---

### Server Management (`server.py`)

**Added:**

- `ServerManager` — complete lifecycle manager for the `llama-server` binary
  - Start, stop, health-check, and restart the HTTP server process
  - Configurable: `host`, `port`, `gpu_layers`, `ctx_size`, `batch_size`, `ubatch_size`
  - Multi-GPU: `split_mode`, `main_gpu`, `tensor_split`, `n_parallel`
  - Performance: `flash_attn`, `cont_batching`, `mlock`, `no_mmap`
  - Auto-download the CUDA binary bundle if not present (via `_internal.bootstrap`)
  - Context manager support for clean startup/shutdown
  - Prometheus `/metrics` endpoint polling

---

### API Client (`api/client.py`)

**Added:**

- `LlamaCppClient` — full HTTP client for the llama-server REST API
  - **Lazy sub-clients** via properties: `.chat`, `.embeddings`, `.models`, `.slots`, `.lora`
  - **ChatCompletionsAPI** — OpenAI-compatible `POST /v1/chat/completions` with streaming
  - **EmbeddingsClientAPI** — `POST /v1/embeddings`, batch support
  - **ModelsClientAPI** — `GET /v1/models`, model metadata
  - **SlotsClientAPI** — multi-slot KV cache management (`GET /slots`, `POST /slots/:id/save`, etc.)
  - **LoraClientAPI** — LoRA adapter hot-swap (`GET /lora-adapters`, `POST /lora-adapters`)
  - **Native completion** with 20+ sampling parameters: Mirostat, DRY, XTC, dynamic temperature, penalty ranges
  - Grammar-guided generation (`grammar`, `json_schema`)
  - SSE streaming with `sseclient`
  - Full response dataclasses: `Message`, `Choice`, `Usage`, `Timings`, `CompletionResponse`, `TokenizeResponse`, etc.

---

### Multi-GPU (`api/multigpu.py`)

**Added:**

- `SplitMode` enum — `NONE`, `LAYER`, `ROW`
- `GPUInfo` dataclass — name, VRAM, compute capability per device
- `MultiGPUConfig` dataclass — unified multi-GPU config for `InferenceEngine` and `ServerManager`
- `detect_gpus()` — enumerate CUDA devices
- `gpu_count()` — count available GPUs
- `get_cuda_version()` — CUDA runtime version
- `get_total_vram()` / `get_free_vram()` — VRAM accounting
- `estimate_model_vram()` — VRAM estimate from model size and quantization
- `can_fit_model()` — VRAM sufficiency check
- `recommend_quantization()` — auto-select GGUF quant type for available VRAM
- `kaggle_t4_dual_config()` — pre-tuned dual-T4 split-layer config
- `colab_t4_single_config()` — single-T4 config
- `auto_config()` — automatic environment-aware config

---

### GGUF Utilities (`api/gguf.py`)

**Added:**

- `GGMLType` — enum for 30+ GGUF quantization types (`Q2_K` through `IQ4_XS`, `F16`, `F32`, `BF16`)
- `GGUFValueType` — metadata value type enum
- `GGUFMetadata`, `GGUFTensorInfo`, `GGUFModelInfo` — parsed model metadata dataclasses
- `quantize(input_path, output_path, quant_type)` — quantize an existing GGUF
- `convert_hf_to_gguf(model_dir, output_path)` — convert HuggingFace checkpoint to GGUF
- `merge_lora(base_model, lora_path, output_path)` — merge LoRA adapter into base GGUF
- `generate_imatrix(model_path, dataset_path)` — generate importance matrix for imatrix quants
- `gguf_report(model_path)` — human-readable GGUF model report
- `report_model_suitability(model_path)` — Kaggle T4 suitability check

---

### NCCL Integration (`api/nccl.py`)

**Added:**

- `NCCLDataType` enum — maps Python types to `ncclDataType_t`
- `NCCLResult` enum — NCCL return codes
- `NCCLCommunicator` — context manager wrapping a NCCL communicator
  - `all_reduce()`, `broadcast()`, `reduce()`, `reduce_scatter()` collective ops
  - Wraps `libnccl.so.2` via ctypes (no PyTorch dependency)
- `is_nccl_available()` / `get_nccl_version()` — NCCL detection
- `get_nccl_info()` / `print_nccl_info()` — diagnostic helpers
- `setup_nccl_environment()` — set recommended NCCL env vars
- `kaggle_nccl_config()` — PCIe T4 preset config

---

### Telemetry (`telemetry/`)

**Added:**

- `setup_telemetry()` — initialize `TracerProvider` + `MeterProvider` with OTLP exporters
- `InferenceTracerProvider` — GPU-aware tracer provider with inference span processors
- `InferenceTracer` — helper for auto-populating `gen_ai.*` span attributes
- `GpuMetricsCollector` — poll GPU utilization, memory, temperature, power via `pynvml`
- `PerformanceMonitor` + `PerformanceSnapshot` / `PerformanceReport` — high-level monitoring context manager
- `InstrumentedLLMClient` — auto-traced wrapper around `LlamaCppClient`
- `LlamaCppClientInstrumentor` — OTel-style monkey-patching instrumentor
- `GraphistryTraceExporter` — OTel `SpanExporter` writing traces to Graphistry graphs
- `semconv.py` — 45 `gen_ai.*` attribute helpers: `set_gen_ai_attr()`, `set_gen_ai_provider()`, `attr_name()`, `metric_name()`
- 5 Gen AI histogram metrics: `gen_ai.client.operation.duration`, `gen_ai.client.token.usage`, `gen_ai.server.request.duration`, `gen_ai.server.time_to_first_token`, `gen_ai.server.time_per_output_token`
- `setup_otlp_env_from_kaggle_secrets()` — load OTLP credentials from Kaggle secrets
- `is_otel_available()` / `is_graphistry_available()` — optional dependency checks

---

### Kaggle Module (`kaggle/`)

**Added:**

- `KaggleEnvironment` — zero-boilerplate environment detection and setup for Kaggle notebooks
  - `is_kaggle()` — detect Kaggle runtime
  - `quick_setup(hf_token, graphistry_token)` — one-call environment initialization
  - GPU detection, binary verification, storage path resolution
- `KaggleSecrets` — load HuggingFace and Graphistry credentials from Kaggle user secrets
- `split_gpu_session()` context manager — dedicate GPU 0 to LLM inference, GPU 1 to visualization
- `ServerPreset` — named server configuration presets for common Kaggle model sizes
- `TensorSplitMode` — enum for Kaggle-specific split strategies
- `KagglePipelineConfig` — full pipeline configuration (server, NCCL, OTLP, Graphistry)
- `GPUContext` — per-GPU context management for split workflows

---

### Inference Optimization (`inference/`)

**Added:**

- `FlashAttentionConfig` — configure FlashAttention v2/v3 (enabled via `flash_attn=True` in `load_model()`)
- `KVCache` / `PagedKVCache` — KV cache configuration and paged memory management
- `ContinuousBatching` — continuous batching settings for multi-slot serving

---

### CUDA Optimization (`cuda/`)

**Added:**

- `CUDAGraph` / `GraphPool` — capture and replay CUDA graphs for inference acceleration
- Triton kernel wrappers for custom attention and matrix ops (`triton_kernels.py`)
- `TensorCoreConfig` — configure TensorCore (FP16/BF16) usage

---

### Quantization (`quantization/`)

**Added:**

- `NF4Quantizer` — NF4 (4-bit NormalFloat) quantization for weight compression
- GGUF conversion utilities (`quantization/gguf.py`) — wrapping `llama-quantize`
- `DynamicQuantizer` — dynamic post-training quantization

---

### Graphistry & RAPIDS (`graphistry/`)

**Added:**

- `GraphistryConnector` — authenticate and upload graphs to Graphistry hub
- RAPIDS `cuGraph` integration for GPU-accelerated graph algorithms
- `GraphWorkload` — graph workload management for split-GPU setups
- Graph builders and visualization helpers

---

### Louie AI (`louie/`)

**Added:**

- `LouieClient` — client for the Louie.ai natural language graph analysis service
- `natural_query()` — query knowledge graphs with natural language
- `KnowledgeExtractor` — extract structured knowledge graphs from LLM output

---

### Unsloth Integration (`unsloth/`)

**Added:**

- Model loader wrapping `unsloth` for efficient 4-bit fine-tuning
- LoRA adapter application and merging
- GGUF export pipeline: fine-tuned model → quantized GGUF → llamatelemetry deployment

---

### Model Registry (`_internal/registry.py`)

**Added:**

30+ curated GGUF models in `MODEL_REGISTRY`, including:

| Family | Sizes |
|--------|-------|
| Gemma 3 (Google) | 1B, 4B, 12B (Q4_K_M, Q8_0) |
| Llama 3.1 / 3.2 (Meta) | 3B, 8B, 70B (Q4_K_M, Q5_K_M) |
| Phi-3.5 / Phi-4 (Microsoft) | 3.8B, 14B (Q4_K_M) |
| Mistral / Mixtral | 7B, 8×7B (Q4_K_M) |
| Qwen 2.5 (Alibaba) | 7B, 14B (Q4_K_M) |
| DeepSeek R1 Distill | 1.5B, 7B (Q4_K_M) |

---

### Bootstrap (`_internal/bootstrap.py`)

**Added:**

- Auto-download T4-optimized binary bundle (~961 MB) from HuggingFace on first import
- GitHub fallback mirror for reliability
- SHA256 integrity verification
- CUDA compute capability check (requires SM 7.5+)
- Platform detection: Kaggle, Colab, local Linux

---

### C++/CUDA Extension (`csrc/`, `llamatelemetry_cpp`)

**Added:**

- `llamatelemetry_cpp` pybind11 module compiled for CUDA 12.x, SM 7.5 (Tesla T4)
- `Device` class — `get_device_count()`, `get_device_properties()`, `set_device()`, `synchronize()`, `get_free_memory()`, `get_total_memory()`
- `Tensor` class — RAII tensor with shape, strides, dtype, device; `.to(device)`, `.cpu()`, `Tensor.zeros()`, `Tensor.ones()`, `Tensor.from_ptr()`
- `DType` enum — `Float32`, `Float16`, `BFloat16`, `Int32`, `Int64`, `UInt8`
- `matmul()` — cuBLAS SGEMM (FP32) and HGEMM (FP16), batched variants
- Static linking: `cudart_static`, `cublas_static`, `cublasLt_static`, `culibos`

---

### Notebooks (18 tutorials)

**Added:**

| # | Notebook | Track |
|---|----------|-------|
| 01 | Quickstart — llamatelemetry v0.1.0 | Foundation |
| 02 | llama-server setup | Foundation |
| 03 | Multi-GPU inference | Foundation |
| 04 | GGUF quantization | Foundation |
| 05 | Unsloth integration | Integration |
| 06 | Split-GPU + Graphistry | Integration |
| 07 | Knowledge graph extraction | Integration |
| 08 | Document network analysis | Integration |
| 09 | Large models on Kaggle | Advanced |
| 10 | Complete workflow | Advanced |
| 11 | GGUF neural network visualization | Advanced |
| 12 | Attention mechanism explorer | Advanced |
| 13 | Token embedding visualizer | Advanced |
| 14 | OpenTelemetry LLM observability | Observability |
| 15 | Real-time performance monitoring | Observability |
| 16 | Production observability | Observability |
| 17 | llamatelemetry + W&B on Kaggle | Observability |
| 18 | OTel + Graphistry trace glue | Observability |

---

### Tests

**Added:**

- 246 passing tests, 24 skipped (CUDA/GPU-only paths)
- `test_llamatelemetry.py` (12,458 lines) — imports, platform detection, GPU compat, binary download, server/engine lifecycle, metrics
- `test_new_apis.py` (7,496 lines) — quantization, Unsloth, CUDA, and inference API coverage
- `test_tensor_api.py` (5,022 lines) — C++ extension: Device, Tensor, matmul, memory management
- `test_gguf_parser.py` (9,438 lines) — GGUF format parser correctness
- `test_full_workflow.py` (1,344 lines) — end-to-end with a real model binary
- `test_end_to_end.py` (3,087 lines) — end-to-end inference test

---

### Documentation

**Added:**

- Full MkDocs Material documentation site: [llamatelemetry.github.io](https://llamatelemetry.github.io/)
- Get Started section: Overview, Installation, Quickstart, Kaggle Quickstart
- 14 Guides: Inference Engine, Server Management, Model Management, API Client, Telemetry, Kaggle, Examples, Graphistry/RAPIDS, Quantization, Unsloth, CUDA Optimizations, Jupyter Workflows, Louie Knowledge Graphs, Troubleshooting
- 13 API Reference pages covering all public APIs
- Notebook Hub with 18 categorized tutorials
- Project section: Architecture, File Map, Release Artifacts, FAQ, Contributing, Changelog

---

### Dependencies (v0.1.0)

**Core (always installed):**

```
numpy>=1.24
requests>=2.28
huggingface_hub>=0.20
tqdm>=4.64
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
```

**Optional (install extras as needed):**

```
opentelemetry-exporter-otlp-proto-http  # OTLP HTTP trace/metrics export
opentelemetry-semantic-conventions      # Gen AI semconv attributes
pygraphistry                            # Graphistry visualization
pandas, matplotlib, scikit-learn        # Data analysis
ipywidgets                              # Jupyter chat widget
torch                                   # Unsloth integration
pynvml                                  # GPU metrics collection
sseclient                               # SSE streaming support
```
