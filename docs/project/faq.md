# Frequently Asked Questions

---

## General

### What is llamatelemetry?

`llamatelemetry` is a CUDA-first Python SDK that makes LLM inference on GGUF models observable, easy to deploy, and deeply integrated with the GPU stack. It orchestrates `llama-server` (the HTTP inference server from llama.cpp) for GGUF model serving, instruments every inference operation with OpenTelemetry tracing and GPU metrics, and provides a high-level Python API that works out-of-the-box on Kaggle T4 x2, Google Colab, and local CUDA Linux machines.

### Why does llamatelemetry exist?

Deploying and observing GGUF LLM inference has a lot of moving parts: downloading the right binary for your CUDA version, configuring multi-GPU splits, wiring up OpenTelemetry spans, and collecting real-time GPU metrics. llamatelemetry unifies all of this into a single Python package with sane defaults so you can go from import to traced inference in a few lines.

### What is the relationship between llamatelemetry and llama.cpp?

llamatelemetry bundles a pre-compiled CUDA-optimized build of `llama-server` (from llama.cpp) and manages its lifecycle. It does not replace or fork llama.cpp — it sits on top of it as a management and observability layer. You can also point llamatelemetry at your own `llama-server` build via `LLAMA_SERVER_PATH`.

### Is this a fork of llama.cpp?

No. llamatelemetry is a pure Python SDK (plus an optional C++/CUDA extension for direct GPU tensor ops) that wraps the llama.cpp HTTP server. The llama.cpp binary is downloaded separately.

---

## Installation

### What are the minimum requirements?

- Python 3.11 or newer
- Linux (primary supported platform)
- CUDA 12.x (for GPU acceleration)
- NVIDIA GPU with compute capability ≥ 7.5 (Turing or newer — Tesla T4, RTX 20xx, RTX 30xx, RTX 40xx, A100, H100)

### Does it run on Windows?

Windows support is limited. The auto-downloaded llama-server binary is Linux-only. If you are on Windows, you can compile llama-server yourself and point `LLAMA_SERVER_PATH` to it, but this is not tested.

### Does it run on macOS?

macOS is not a supported target. llamatelemetry is optimized for NVIDIA CUDA; Apple Silicon (MLX/Metal) is out of scope for v0.1.1.

### Does it run without a GPU?

Yes, llamatelemetry will import and run on CPU-only machines, but performance will be slow (no GPU offloading). CUDA-specific features (GPU metrics collection, NCCL, CUDA graphs, TensorCore) will gracefully return `None` or be disabled. The OpenTelemetry tracing layer works regardless of GPU availability.

### How do I install it?

```bash
pip install llamatelemetry
```

For Kaggle notebooks, see the [Kaggle Quickstart](../get-started/kaggle-quickstart.md).

For source installation with the C++/CUDA extension:

```bash
git clone https://github.com/llamatelemetry/llamatelemetry.git
cd llamatelemetry
pip install -e .
```

### What optional extras are available?

```bash
pip install llamatelemetry[otel]        # OTLP exporters for traces/metrics
pip install llamatelemetry[graphistry]  # Graphistry graph visualization
pip install llamatelemetry[jupyter]     # Jupyter chat widget + ipywidgets
pip install llamatelemetry[unsloth]     # Fine-tuning with Unsloth + LoRA
pip install llamatelemetry[all]         # All optional extras
```

---

## Binary Download and Bootstrap

### What gets downloaded on first import?

On the first `import llamatelemetry`, the bootstrap layer checks for `llama-server`. If it is not found, it downloads a T4-optimized binary bundle (~961 MB) from HuggingFace. This download only happens once; the binary is cached in the package directory.

### Where is the binary cached?

The binary is cached in `~/.cache/llamatelemetry/` or in the package installation directory. You can inspect the path via:

```python
from llamatelemetry._internal.bootstrap import get_binary_path
print(get_binary_path())
```

### How do I use my own llama-server build?

Set the environment variable before importing:

```bash
export LLAMA_SERVER_PATH=/path/to/your/llama-server
```

Or pass it explicitly:

```python
from llamatelemetry import ServerManager
server = ServerManager(llama_server_path="/path/to/llama-server")
```

### The download is failing — what should I try?

1. Check your internet connection and HuggingFace access
2. Try the GitHub fallback mirror by setting `LLAMATELEMETRY_MIRROR=github`
3. Download the bundle manually from the [releases page](release-artifacts.md) and set `LLAMA_SERVER_PATH`
4. On Kaggle, use the [Kaggle Quickstart](../get-started/kaggle-quickstart.md) notebook which pre-stages the binary via a dataset

---

## Models

### What model formats are supported?

llamatelemetry supports all GGUF models compatible with llama.cpp. This includes Q2_K through IQ4_XS quantization types as well as F16 and F32 (where VRAM allows). Non-GGUF formats (GGML v1/v2, PyTorch `.bin`, SafeTensors) are not directly supported but can be converted to GGUF via `llamatelemetry.api.gguf.convert_hf_to_gguf()`.

### Where are models stored by default?

Models downloaded via the registry or `SmartModelDownloader` are stored in `~/.cache/llamatelemetry/models/` or the path returned by `llamatelemetry.get_models_dir()`. You can also pass an absolute local path to `load_model()`.

### How do I load a model from HuggingFace?

```python
engine.load_model("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
```

Use the `repo_id:filename` syntax for any HuggingFace GGUF model.

### How do I load a local GGUF file?

```python
engine.load_model("/path/to/my-model-Q4_K_M.gguf")
```

### What is the MODEL_REGISTRY?

`MODEL_REGISTRY` is a curated dictionary of 30+ well-tested GGUF models with known-good quantizations for T4 hardware. Use a registry key (e.g., `"gemma-3-4b-Q4_K_M"`) with `load_model()` for the simplest experience — it handles download, verification, and launch automatically.

### Which model should I use for a 16 GB GPU?

For a single T4 (16 GB):

| Model Size | Recommended Quant | Context Size |
|-----------|-------------------|--------------|
| 3–4B | Q8_0 | 8192 |
| 7–8B | Q4_K_M | 4096 |
| 13B | Q4_K_M | 2048 |

For dual T4 (32 GB combined, LAYER split):

| Model Size | Recommended Quant | Context Size |
|-----------|-------------------|--------------|
| 13–14B | Q4_K_M | 8192 |
| 30–34B | Q4_K_M | 4096 |
| 70B | Q2_K | 2048 |

Use `recommend_quantization(model_size_b, available_vram_gb)` for an automatic recommendation.

---

## Inference

### How do I run basic inference?

```python
import llamatelemetry

with llamatelemetry.InferenceEngine() as engine:
    engine.load_model("gemma-3-1b-Q4_K_M")
    result = engine.infer("What is the capital of France?")
    print(result.text)
```

### How do I stream tokens as they are generated?

```python
for token in engine.infer_stream("Write a haiku about CUDA:"):
    print(token, end="", flush=True)
```

### How do I run batch inference?

```python
prompts = ["Summarize AI in 3 words", "What is GGUF?", "Explain CUDA graphs"]
results = engine.batch_infer(prompts, max_tokens=128)
for r in results:
    print(r.text)
```

### How do I do multi-turn chat?

Use `ChatEngine` for conversation management:

```python
from llamatelemetry.chat import ChatEngine

chat = ChatEngine(engine)
chat.add_system("You are a helpful assistant.")
response = chat.send("What is llamatelemetry?")
print(response)
response2 = chat.send("Can you give me a code example?")
print(response2)
```

### How do I generate embeddings?

```python
from llamatelemetry.embeddings import EmbeddingEngine

emb_engine = EmbeddingEngine(engine)
vectors = emb_engine.embed(["Hello world", "CUDA inference"])
print(vectors.shape)  # (2, embedding_dim)
```

### What sampling parameters are available?

Through `LlamaCppClient`, you have access to 20+ sampling parameters including:
`temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `frequency_penalty`, `presence_penalty`, `seed`, `mirostat`, `mirostat_tau`, `mirostat_eta`, `dynatemp_range`, `dynatemp_exponent`, `dry_multiplier`, `dry_base`, `xtc_threshold`, `xtc_probability`, `grammar`, `json_schema`.

---

## Multi-GPU

### Does llamatelemetry support multi-GPU inference?

Yes. Pass a `MultiGPUConfig` to `load_model()`:

```python
from llamatelemetry.api.multigpu import kaggle_t4_dual_config

config = kaggle_t4_dual_config(model_size_b=13.0)
engine.load_model("model-name-Q4_K_M", multi_gpu_config=config)
```

### What split modes are available?

- `SplitMode.LAYER` — distribute transformer layers across GPUs (recommended for PCIe setups like Kaggle T4 x2)
- `SplitMode.ROW` — tensor-parallel row split (requires NVLink for efficiency; not recommended on Kaggle)
- `SplitMode.NONE` — single GPU

### Does it require NVLink for multi-GPU?

No. `SplitMode.LAYER` works efficiently over PCIe, which is what Kaggle T4 x2 uses. `SplitMode.ROW` benefits from NVLink but is not required for layer splitting.

### Does it support more than 2 GPUs?

The architecture supports N GPUs but the primary test target is dual-T4. Three or more GPUs should work with `MultiGPUConfig(n_gpu=N, tensor_split=[...])` but this configuration is less tested.

---

## Telemetry & Observability

### Is OpenTelemetry required?

No. Telemetry is fully optional. If `opentelemetry-api` and `opentelemetry-sdk` are not installed, all telemetry features are silently disabled and `is_otel_available()` returns `False`. Inference works identically with or without telemetry.

### What does llamatelemetry trace?

Every inference call can be wrapped in an OpenTelemetry span with full `gen_ai.*` semantic convention attributes: model name, provider, operation type, temperature, token counts (input/output), finish reasons, session ID, and more. GPU metrics (utilization, memory, temperature, power) are collected separately via `GpuMetricsCollector`.

### How do I send traces to a backend (Jaeger, Grafana, Honeycomb, etc.)?

```python
from llamatelemetry.telemetry import setup_telemetry

setup_telemetry(
    service_name="my-llm-service",
    otlp_endpoint="https://otlp.example.com/v1/traces",
    otlp_headers={"Authorization": "Bearer my-token"},
)
```

Any OTLP-compatible backend works: Jaeger, Grafana Tempo, Honeycomb, Lightstep, Datadog, New Relic, etc.

### How do I set up telemetry on Kaggle?

Add your `OTLP_ENDPOINT` and `OTLP_TOKEN` to Kaggle user secrets, then:

```python
from llamatelemetry.telemetry import setup_otlp_env_from_kaggle_secrets, setup_telemetry

env = setup_otlp_env_from_kaggle_secrets()
setup_telemetry(
    service_name="kaggle-inference",
    otlp_endpoint=env.get("endpoint"),
    otlp_headers={"Authorization": f"Bearer {env.get('token', '')}"},
)
```

### What are the 5 Gen AI metrics?

| Metric | Unit | What it measures |
|--------|------|-----------------|
| `gen_ai.client.operation.duration` | s | End-to-end latency (client side) |
| `gen_ai.client.token.usage` | {token} | Input and output token counts |
| `gen_ai.server.request.duration` | s | Server-side generation time |
| `gen_ai.server.time_to_first_token` | s | Prefill latency (TTFT) |
| `gen_ai.server.time_per_output_token` | s | Decode step latency (TPOT) |

---

## Kaggle

### What is the recommended Kaggle setup?

1. Add your HuggingFace token as a Kaggle secret (`HF_TOKEN`)
2. Enable GPU Accelerator: T4 x2
3. Use the [Kaggle Quickstart](../get-started/kaggle-quickstart.md)

### Does it work on Kaggle without internet?

No. Kaggle's internet-off mode prevents model downloads and OTLP export. Keep internet enabled for llamatelemetry notebooks.

### What is `split_gpu_session()`?

`split_gpu_session()` is a context manager that sets GPU visibility so that GPU 0 is reserved for LLM inference and GPU 1 is reserved for Graphistry/RAPIDS visualization. This prevents VRAM contention on dual-T4 setups.

```python
from llamatelemetry.kaggle import split_gpu_session

with split_gpu_session() as ctx:
    # ctx.inference_gpu == 0, ctx.viz_gpu == 1
    engine.load_model("model-Q4_K_M", multi_gpu_config=ctx.inference_config)
```

### Why is the first inference slow on Kaggle?

The first call involves:
1. llama-server startup (a few seconds)
2. Model loading from disk into VRAM (depends on model size and storage speed)
3. CUDA kernel compilation warmup on first forward pass

Subsequent inferences are fast.

---

## Performance

### How do I check tokens per second?

```python
result = engine.infer("Hello, world!")
print(f"{result.tokens_per_sec:.1f} tok/s")
```

Or use `PerformanceMonitor` for sustained monitoring:

```python
from llamatelemetry.telemetry.monitor import PerformanceMonitor

with PerformanceMonitor(gpu_indices=[0, 1]) as monitor:
    result = engine.infer("...")
report = monitor.report()
print(f"Avg: {report.avg_tokens_per_second:.1f} tok/s")
```

### How do I improve inference speed?

1. **Use FlashAttention:** `engine.load_model("...", flash_attn=True)`
2. **Maximize GPU layers:** Set `gpu_layers` to the full model layer count
3. **Use continuous batching for multiple requests:** `n_parallel=2` or higher
4. **Choose the right quantization:** Q4_K_M is usually the best speed/quality balance
5. **Increase batch sizes:** `batch_size=512, ubatch_size=512`
6. **Use mlock:** `mlock=True` to lock model weights in RAM

### What is the typical throughput on Kaggle T4?

Rough benchmarks on Kaggle T4 x2:

| Model | Quant | Split | Tokens/sec |
|-------|-------|-------|-----------|
| Gemma 3 1B | Q4_K_M | Single | ~80–120 |
| Gemma 3 4B | Q4_K_M | Single | ~35–55 |
| Llama 3.1 8B | Q4_K_M | Single | ~18–25 |
| Llama 3.1 8B | Q4_K_M | Dual | ~28–40 |
| Llama 3.1 70B | Q4_K_M | Dual | ~3–5 |

These numbers depend on context length, batch size, and temperature.

---

## Errors and Troubleshooting

### `ImportError: No module named 'llamatelemetry_cpp'`

The C++/CUDA extension was not built or is not on the Python path. Either:
- Install the CUDA binary release: download from the [releases page](release-artifacts.md)
- Build from source: `pip install -e .` (requires CUDA toolkit and CMake)

The pure Python functionality works without the extension; only direct C++ tensor ops require `llamatelemetry_cpp`.

### `ConnectionError: llama-server is not responding`

The llama-server process failed to start. Common causes:
- The GGUF model file is corrupted or truncated — re-download it
- Not enough VRAM — reduce `gpu_layers` or use a smaller quantization
- Binary incompatibility — download the CUDA 12.x binary for your GPU architecture
- Port conflict — another process is using port 8090. Change it with `server_url="http://127.0.0.1:8091"`

### `OutOfMemoryError` or server crashes after loading

- Reduce `gpu_layers` by 10–20% to leave VRAM headroom
- Shrink context size: `ctx_size=2048` instead of 4096
- Switch to a lower quantization (e.g., Q4_K_M → Q3_K_M)
- Enable `mmap=True` to use system RAM as overflow

### Telemetry spans are not appearing in my backend

1. Check that `otlp_endpoint` is correct and reachable from your network
2. Verify authentication headers
3. Enable console export for debugging: `setup_telemetry(enable_console_export=True)`
4. Check that `tracer_provider.shutdown()` is called before the process exits (flushes buffered spans)
5. On Kaggle, ensure internet access is enabled

### NCCL errors on Kaggle T4 x2

Kaggle T4 x2 uses PCIe (no NVLink). Run:

```python
from llamatelemetry.api.nccl import setup_nccl_environment
setup_nccl_environment(disable_p2p=True, disable_ib=True)
```

This disables peer-to-peer and InfiniBand transport, forcing NCCL to use socket-based communication which works on Kaggle.

### `CUDA error: device-side assert triggered`

This usually means a tensor operation received an out-of-range index. Causes:
- Tokenizer mismatch (wrong tokenizer for the model)
- Corrupted model weights — re-download the GGUF
- Context length exceeded — reduce prompt length or increase `ctx_size`

---

## Architecture

### How does the auto-bootstrap work?

When you `import llamatelemetry`, `_internal/bootstrap.py` runs and:
1. Checks if `llama-server` is already present in the cache
2. If not, downloads the T4-optimized binary bundle from HuggingFace (primary) or GitHub (fallback)
3. Verifies SHA256 integrity
4. Checks CUDA compute capability (requires ≥ SM 7.5)

The download is ~961 MB and only happens once.

### What is the `llamatelemetry_cpp` module?

It is a pybind11 C++ extension that exposes a `Device` class (CUDA device management), a `Tensor` class (RAII GPU tensors), and cuBLAS `matmul()` operations directly from Python. It is used for direct GPU tensor operations and benchmarking without going through PyTorch or other frameworks.

### Why does llamatelemetry bundle llama-server instead of using the system llama.cpp?

To guarantee binary compatibility with CUDA 12.x and SM 7.5. The bundled binary is pre-compiled with exactly the right CUDA flags and optimizations for the T4 target. You can always override this with `LLAMA_SERVER_PATH`.

### Is llamatelemetry compatible with llama.cpp's OpenAI API spec?

Yes. `LlamaCppClient` implements the full OpenAI-compatible REST API as served by llama-server, including `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`, streaming SSE, and native completions. It also exposes llama.cpp-specific endpoints like `/slots`, `/lora-adapters`, and `/metrics`.
