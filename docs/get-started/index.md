# Get Started

Welcome to **llamatelemetry**, a CUDA-first OpenTelemetry SDK for LLM inference
observability. This section walks you through installation, your first inference
call, and the Kaggle-optimized workflow, so you can move from zero to running
GPU-accelerated language models in minutes.

---

## What is llamatelemetry?

llamatelemetry is a Python orchestration layer built on top of
[llama.cpp](https://github.com/ggerganov/llama.cpp). It automates binary
bootstrapping, model discovery, server lifecycle management, inference requests,
and OpenTelemetry-based observability---all from a single `pip install`.

The SDK targets NVIDIA GPUs with CUDA 12.x and is production-tested on Tesla T4
hardware (SM 7.5), including Kaggle dual-T4 notebook environments. It exposes a
high-level `InferenceEngine` API for quick prototyping and a lower-level
`LlamaCppClient` for full control over the OpenAI-compatible llama-server REST
interface.

### Who is it for?

- **ML engineers** who need fast, quantized LLM inference on consumer or cloud
  GPUs without writing C++ code.
- **Platform teams** who want standardized OpenTelemetry traces and metrics
  (following the Gen AI semantic conventions) across inference workloads.
- **Kaggle competitors** and **notebook authors** who need reproducible,
  GPU-aware pipelines with minimal boilerplate.
- **Researchers** exploring quantization, fine-tuning (Unsloth/LoRA), or
  graph-based knowledge extraction workflows.

---

## SDK architecture at a glance

llamatelemetry ships approximately 40 Python source files and 7 C++/CUDA files,
organized into 10 modules. Each module is self-contained and imports only what it
needs, so you can use the inference engine without ever touching the graph
visualization layer, and vice versa.

| Module | Purpose |
|---|---|
| **api** | `LlamaCppClient` (OpenAI-compatible + native llama.cpp API), `MultiGPUConfig` (split modes, GPU detection, presets), GGUF utilities, NCCL collective operations |
| **telemetry** | OpenTelemetry instrumentation: 45 `gen_ai.*` span attributes, 5 metrics instruments, auto-instrumentation hooks, OTLP exporters |
| **kaggle** | `KaggleEnvironment` detection, `ServerPreset` enum, `split_gpu_session` context manager, Kaggle secrets integration |
| **inference** | FlashAttention configuration, KV cache management, continuous batching helpers |
| **cuda** | CUDAGraph capture/replay, Triton kernel launchers, TensorCore utilities |
| **quantization** | NF4 quantization, GGUF format conversion, dynamic quantization policies |
| **graphistry** | Graph visualization with Graphistry and RAPIDS cuGraph integration |
| **louie** | AI-driven graph analysis, knowledge extraction pipelines |
| **unsloth** | Fine-tuning orchestration, LoRA adapter management, GGUF export |
| **_internal** | Bootstrap logic (auto-downloads ~961 MB of binaries on first import), `MODEL_REGISTRY` with 22+ curated model entries |

### C++/CUDA extension

The `llamatelemetry_cpp` pybind11 module provides device operations, a Tensor
RAII wrapper supporting 6 data types, and cuBLAS matrix multiplication kernels
(SGEMM and HGEMM). It links against `cudart_static`, `cublas_static`, and
`cublasLt_static`.

---

## System requirements

### Minimum

| Component | Requirement |
|---|---|
| Python | >= 3.11 |
| OS | Linux (Ubuntu 20.04+ recommended) |
| NVIDIA driver | >= 525.x |
| CUDA toolkit | 12.x |
| GPU | Any NVIDIA GPU with compute capability >= 6.1 |
| RAM | 8 GB system memory |
| Disk | 2 GB free (binaries + one small model) |

### Recommended

| Component | Recommendation |
|---|---|
| GPU | Tesla T4 (16 GB VRAM, SM 7.5) or better |
| VRAM | 16 GB per GPU for 7B-parameter models at Q4 quantization |
| Disk | 10 GB+ free for multiple model downloads |
| Network | Broadband for first-run bootstrap and model downloads |

### Kaggle

Kaggle notebooks with the **GPU T4 x2** accelerator are the primary tested
environment. Each T4 provides 16 GB VRAM and SM 7.5 compute capability. The SDK
includes presets (`KAGGLE_DUAL_T4`, `KAGGLE_SINGLE_T4`) that automatically
configure server parameters, GPU layer counts, and context sizes for these
machines.

---

## The shortest path from install to inference

The complete workflow has five steps:

1. **Install** the SDK from GitHub with `pip`.
2. **Verify** that CUDA and your GPU are visible to the runtime.
3. **Create** an `InferenceEngine` instance.
4. **Load** a model from the built-in registry (or a local GGUF file).
5. **Run** inference and inspect the `InferResult`.

```python
#Step 1: Install llamatelemetry
!pip install -q --no-cache-dir --force-reinstall git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1

import llamatelemetry as lt
print("\nllamatelemetry version:", llamatelemetry.__version__)

# Step 2: verify CUDA
cuda_info = lt.detect_cuda()
print(f"CUDA available: {cuda_info['available']}")
for gpu in cuda_info["gpus"]:
    print(f"  {gpu['name']} - {gpu['memory']} MB")

# Step 3: create engine
engine = lt.InferenceEngine(enable_telemetry=False)

# Step 4: load model (downloads on first run)
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)

# Step 5: run inference
result = engine.infer("Explain GPU tensor cores in two sentences.")
print(result.text)
print(f"Tokens/sec: {result.tokens_per_sec:.1f}")
```

After you have seen a successful result, continue to the detailed guides for
model management, telemetry configuration, multi-GPU inference, and more.

---

## Core dependencies

llamatelemetry keeps its required dependency footprint small:

| Package | Role |
|---|---|
| `numpy` | Numerical operations and tensor utilities |
| `requests` | HTTP communication with llama-server |
| `huggingface_hub` | Model downloads from Hugging Face |
| `tqdm` | Progress bars for downloads and batch operations |
| `opentelemetry-api` | Telemetry span and metric API |
| `opentelemetry-sdk` | Telemetry SDK (exporters, processors) |

Optional dependency groups (installable via extras) add support for OTLP
exporters, Graphistry visualization, pandas DataFrames, Jupyter widgets, PyTorch,
pynvml GPU monitoring, SSE streaming, and Weights & Biases logging.

---

## Choose your path

Depending on your environment and goals, start with the page that matches best:

### Local workstation or cloud VM

Follow the standard installation and quickstart:

1. [Installation](installation.md) -- system prerequisites, pip install,
   environment variables, optional extras, and troubleshooting.
2. [Quickstart](quickstart.md) -- end-to-end tutorial from GPU verification
   through batch inference, streaming, the low-level client API, chat
   completions, embeddings, and cleanup.

### Kaggle notebook (T4 x2)

Jump directly to the Kaggle-specific guide:

1. [Kaggle Quickstart](kaggle-quickstart.md) -- notebook setup, `ServerPreset`
   configuration, `split_gpu_session`, OTLP secrets, the full Kaggle pipeline,
   and recommended models for T4 VRAM budgets.

### Deep dives after getting started

Once you are running inference, explore these next:

- [Inference Engine Guide](../guides/inference-engine.md) -- advanced
  `InferenceEngine` usage, context management, error handling.
- [Server Management](../guides/server-management.md) -- `ServerManager`
  lifecycle, health checks, port configuration.
- [Model Management](../guides/model-management.md) -- the model registry,
  Hugging Face downloads, GGUF format details.
- [Telemetry and Observability](../guides/telemetry-observability.md) --
  OpenTelemetry setup, 45 Gen AI attributes, Grafana dashboards.
- [API Client Reference](../guides/api-client.md) -- `LlamaCppClient` for
  chat completions, embeddings, tokenization.
- [Kaggle Environment Guide](../guides/kaggle-environment.md) -- advanced
  Kaggle patterns, secrets management, GPU splitting.
- [Notebook Hub](../notebooks/index.md) -- 18 production-tested Kaggle
  notebooks covering every major workflow.

---

## Getting help

- **FAQ**: [Frequently Asked Questions](../project/faq.md)
- **Troubleshooting**: [Common issues and fixes](../guides/troubleshooting.md)
- **GitHub Issues**: [github.com/llamatelemetry/llamatelemetry/issues](https://github.com/llamatelemetry/llamatelemetry/issues)
- **Changelog**: [Release history](../project/changelog.md)
