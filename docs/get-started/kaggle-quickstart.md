# Kaggle Quickstart

This guide is purpose-built for **Kaggle notebooks with GPU T4 x2** accelerator.
llamatelemetry includes Kaggle-specific presets, pipeline helpers, secrets
integration, and a split-GPU context manager that dedicates one T4 to inference
and the other to analytics workloads like Graphistry visualization.

Every code block below is designed to run as a Kaggle notebook cell.

---

## 1. Enable GPU accelerator

Before running any code, configure the Kaggle notebook:

1. Open your notebook in Kaggle.
2. Click **Settings** (right sidebar) or the three-dot menu.
3. Under **Accelerator**, select **GPU T4 x2**.
4. Save and restart the session.

Both Tesla T4 GPUs provide 16 GB VRAM each (SM 7.5, compute capability 7.5).

---

## 2. Install llamatelemetry

Run this as the first code cell. The `-q` flag suppresses pip output to keep the
notebook clean:

```python
!pip -q install --no-cache-dir --force-reinstall \
  git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1
```

For optional dependencies needed by specific workflows:

```python
# Telemetry export to Grafana Cloud or other OTLP backends
!pip -q install opentelemetry-exporter-otlp-proto-http

# Graphistry visualization (requires Graphistry API key)
!pip -q install pygraphistry pandas

# Weights & Biases logging
!pip -q install wandb

# SSE streaming support
!pip -q install sseclient-py
```

---

## 3. Verify CUDA and GPU visibility

Confirm both T4 GPUs are visible to the SDK:

```python
import llamatelemetry as lt

cuda_info = lt.detect_cuda()
print(f"CUDA available: {cuda_info['available']}")
print(f"CUDA version:   {cuda_info['version']}")
print(f"GPU count:       {len(cuda_info['gpus'])}")

for i, gpu in enumerate(cuda_info["gpus"]):
    print(f"\n  GPU {i}: {gpu['name']}")
    print(f"    VRAM:               {gpu['memory']} MB")
    print(f"    Driver:             {gpu['driver_version']}")
    print(f"    Compute capability: {gpu['compute_capability']}")
```

Expected output:

```
CUDA available: True
CUDA version:   12.2
GPU count:       2

  GPU 0: Tesla T4
    VRAM:               15360 MB
    Driver:             535.104.05
    Compute capability: 7.5

  GPU 1: Tesla T4
    VRAM:               15360 MB
    Driver:             535.104.05
    Compute capability: 7.5
```

---

## 4. ServerPreset for Kaggle

The `ServerPreset` enum provides pre-tuned configurations for common GPU
environments. Each preset sets appropriate values for GPU layers, context size,
number of parallel slots, and split mode:

```python
from llamatelemetry.kaggle import ServerPreset

# Available presets
print(ServerPreset.KAGGLE_DUAL_T4)    # Dual T4 with layer splitting
print(ServerPreset.KAGGLE_SINGLE_T4)  # Single T4, all layers on GPU 0
print(ServerPreset.COLAB_T4)          # Google Colab T4 environment
```

### Start a server from a preset

The `start_server_from_preset()` function creates a `ServerManager`, applies
the preset configuration, loads the specified model, and starts the server in
one call:

```python
from llamatelemetry.kaggle import start_server_from_preset, ServerPreset

server = start_server_from_preset(
    preset=ServerPreset.KAGGLE_DUAL_T4,
    model_name="gemma-3-1b-Q4_K_M",
)

print(f"Server running at: {server.server_url}")
print(f"Server PID:        {server.pid}")
```

The dual-T4 preset splits model layers across both GPUs, effectively doubling
the available VRAM for larger models. For models that fit on a single GPU, use
`KAGGLE_SINGLE_T4` to keep GPU 1 free for analytics.

---

## 5. Basic inference on Kaggle

Once the server is running, create an `InferenceEngine` and run inference:

```python
engine = lt.InferenceEngine(
    server_url="http://127.0.0.1:8080",
    enable_telemetry=False,
)

engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)

result = engine.infer(
    "Summarize the benefits of running LLMs on Kaggle T4 GPUs.",
    max_tokens=128,
    temperature=0.7,
)

print(result.text)
print(f"\nTokens generated: {result.tokens_generated}")
print(f"Latency:          {result.latency_ms:.0f} ms")
print(f"Throughput:       {result.tokens_per_sec:.1f} tokens/sec")
```

---

## 6. Split-GPU session

The `split_gpu_session` context manager assigns GPUs to different workloads.
This is the recommended pattern when you need inference on one GPU and
analytics (Graphistry, cuGraph, or other CUDA workloads) on the other:

```python
from llamatelemetry.kaggle import split_gpu_session

with split_gpu_session(llm_gpu=0, graph_gpu=1):
    # GPU 0: inference workload
    engine = lt.InferenceEngine(enable_telemetry=False)
    engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)

    result = engine.infer("What is GPU memory splitting?", max_tokens=96)
    print(result.text)

    # GPU 1 is available for Graphistry, cuGraph, or other CUDA work
    # Example: pygraphistry visualization would use GPU 1 here

    engine.unload_model()
```

Inside the context, `CUDA_VISIBLE_DEVICES` is configured so that the LLM
server sees only `llm_gpu` and analytics libraries see only `graph_gpu`. When
the context exits, the original GPU visibility is restored.

---

## 7. KagglePipelineConfig

For more complex setups, `KagglePipelineConfig` bundles all configuration into a
single object:

```python
from llamatelemetry.kaggle import KagglePipelineConfig

config = KagglePipelineConfig(
    model_name="gemma-3-1b-Q4_K_M",
    preset=ServerPreset.KAGGLE_DUAL_T4,
    enable_telemetry=True,
    service_name="kaggle-llm-experiment",
    llm_gpu=0,
    graph_gpu=1,
)

print(f"Model:     {config.model_name}")
print(f"Preset:    {config.preset}")
print(f"Telemetry: {config.enable_telemetry}")
print(f"LLM GPU:   {config.llm_gpu}")
print(f"Graph GPU: {config.graph_gpu}")
```

Use the config to drive the full pipeline:

```python
from llamatelemetry.kaggle import start_server_from_preset

server = start_server_from_preset(
    preset=config.preset,
    model_name=config.model_name,
)

engine = lt.InferenceEngine(
    enable_telemetry=config.enable_telemetry,
    telemetry_config={
        "service_name": config.service_name,
        "enable_llama_metrics": True,
    },
)

engine.load_model(config.model_name, auto_start=False)  # server already running
result = engine.infer("Explain KV cache eviction.", max_tokens=96)
print(result.text)
```

---

## 8. OTLP telemetry from Kaggle secrets

Kaggle provides a secrets store for API keys and credentials. The
`load_grafana_otlp_env_from_kaggle()` function reads OTLP endpoint and
authentication credentials from Kaggle secrets and sets the corresponding
environment variables:

```python
from llamatelemetry.kaggle import load_grafana_otlp_env_from_kaggle

# Reads from Kaggle secrets:
#   GRAFANA_OTLP_ENDPOINT
#   GRAFANA_OTLP_TOKEN
# and sets:
#   OTEL_EXPORTER_OTLP_ENDPOINT
#   OTEL_EXPORTER_OTLP_HEADERS
load_grafana_otlp_env_from_kaggle()
```

### Setting up Kaggle secrets

Before using this function, add the following secrets in your Kaggle notebook
settings:

1. **GRAFANA_OTLP_ENDPOINT** -- Your Grafana Cloud OTLP endpoint
   (e.g., `https://otlp-gateway-prod-us-central-0.grafana.net/otlp`).
2. **GRAFANA_OTLP_TOKEN** -- A base64-encoded `instance_id:api_key` string for
   Grafana Cloud authentication.

To add secrets: **Settings** (right sidebar) > **Secrets** > **Add a new
secret**.

---

## 9. Full OTEL + client setup

The `setup_otel_and_client()` helper combines telemetry initialization and
client creation into a single call:

```python
from llamatelemetry.kaggle import (
    load_grafana_otlp_env_from_kaggle,
    setup_otel_and_client,
    start_server_from_preset,
    ServerPreset,
)

# Step 1: Load OTLP credentials from Kaggle secrets
load_grafana_otlp_env_from_kaggle()

# Step 2: Start the server
server = start_server_from_preset(
    preset=ServerPreset.KAGGLE_DUAL_T4,
    model_name="gemma-3-1b-Q4_K_M",
)

# Step 3: Initialize OTEL and create an instrumented client
otel_client = setup_otel_and_client(
    service_name="kaggle-observability-demo",
    server_url="http://127.0.0.1:8080",
)

# Now all requests through otel_client emit traces and metrics
result = otel_client.chat_completions(
    messages=[{"role": "user", "content": "What is OpenTelemetry?"}],
    max_tokens=128,
)
print(result["choices"][0]["message"]["content"])
```

Traces and metrics are automatically exported to the configured OTLP endpoint.
View them in Grafana Cloud dashboards or any OTLP-compatible backend.

---

## 10. Weights & Biases integration

For experiment tracking, combine llamatelemetry with Weights & Biases:

```python
import wandb

# Initialize W&B (reads WANDB_API_KEY from Kaggle secrets)
wandb.init(
    project="llamatelemetry-kaggle",
    config={
        "model": "gemma-3-1b-Q4_K_M",
        "max_tokens": 128,
        "temperature": 0.7,
        "gpu": "T4 x2",
    },
)

engine = lt.InferenceEngine(enable_telemetry=False)
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)

prompts = [
    "Explain flash attention.",
    "What is model quantization?",
    "How does continuous batching work?",
]

for prompt in prompts:
    result = engine.infer(prompt, max_tokens=128, temperature=0.7)

    # Log each inference result to W&B
    wandb.log({
        "prompt": prompt,
        "tokens_generated": result.tokens_generated,
        "latency_ms": result.latency_ms,
        "tokens_per_sec": result.tokens_per_sec,
        "success": result.success,
    })

    print(f"{result.tokens_per_sec:.1f} tok/s | {prompt[:50]}")

engine.unload_model()
wandb.finish()
```

Add `WANDB_API_KEY` to Kaggle secrets for automatic authentication.

---

## 11. Recommended models for T4 VRAM

Each Tesla T4 has 16 GB VRAM. The following table shows recommended models from
the built-in registry and their approximate VRAM requirements at Q4_K_M
quantization:

| Model | Parameters | VRAM (approx.) | Fits on | Notes |
|---|---|---|---|---|
| `gemma-3-1b-Q4_K_M` | 1B | ~1.5 GB | Single T4 | Fast prototyping, default small model |
| `qwen2.5-1.5b-Q4_K_M` | 1.5B | ~2 GB | Single T4 | Good instruction following |
| `gemma-2-2b-Q4_K_M` | 2B | ~2.5 GB | Single T4 | Balanced quality/speed |
| `phi-3-mini-Q4_K_M` | 3.8B | ~3 GB | Single T4 | Strong reasoning for size |
| `llama-3.2-3b-Q4_K_M` | 3B | ~2.5 GB | Single T4 | Meta Llama family |
| `mistral-7b-Q4_K_M` | 7B | ~5 GB | Single T4 | Strong general-purpose |
| `llama-3.1-8b-Q4_K_M` | 8B | ~6 GB | Single T4 | Excellent quality |
| `gemma-2-9b-Q4_K_M` | 9B | ~7 GB | Single T4 | High quality, fits T4 |
| `llama-3.1-70b-Q4_K_M` | 70B | ~40 GB | Dual T4 (split) | Requires layer splitting across both GPUs |

For models larger than 16 GB, use `ServerPreset.KAGGLE_DUAL_T4` to split layers
across both GPUs. The 70B model at Q4 quantization requires both T4s and leaves
minimal headroom, so use a smaller context size.

### Choosing a model

- **Quick experiments**: `gemma-3-1b-Q4_K_M` (fastest load, minimal VRAM).
- **Quality inference**: `llama-3.1-8b-Q4_K_M` or `gemma-2-9b-Q4_K_M`.
- **Split-GPU with analytics**: Use a single-T4 model and keep GPU 1 free for
  Graphistry or cuGraph.
- **Maximum model size**: `llama-3.1-70b-Q4_K_M` with dual-T4 splitting (tight
  VRAM budget, reduce `ctx_size`).

---

## 12. Complete Kaggle notebook example

This cell-by-cell example combines everything into a production-ready Kaggle
workflow:

### Cell 1: Install

```python
!pip -q install --no-cache-dir --force-reinstall \
  git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1
!pip -q install opentelemetry-exporter-otlp-proto-http
```

### Cell 2: Verify environment

```python
import llamatelemetry as lt

cuda_info = lt.detect_cuda()
assert cuda_info["available"], "CUDA not available"
assert len(cuda_info["gpus"]) >= 2, "Need 2 GPUs for dual-T4 preset"
print(f"Ready: {len(cuda_info['gpus'])} x {cuda_info['gpus'][0]['name']}")

lt.setup_environment()
```

### Cell 3: Configure pipeline

```python
from llamatelemetry.kaggle import (
    KagglePipelineConfig,
    ServerPreset,
    load_grafana_otlp_env_from_kaggle,
    split_gpu_session,
)

# Load telemetry credentials (optional, skip if not using Grafana)
# load_grafana_otlp_env_from_kaggle()

config = KagglePipelineConfig(
    model_name="gemma-3-1b-Q4_K_M",
    preset=ServerPreset.KAGGLE_DUAL_T4,
    enable_telemetry=False,
    llm_gpu=0,
    graph_gpu=1,
)
```

### Cell 4: Run inference

```python
with split_gpu_session(llm_gpu=config.llm_gpu, graph_gpu=config.graph_gpu):
    engine = lt.InferenceEngine(enable_telemetry=config.enable_telemetry)
    engine.load_model(config.model_name, auto_start=True)

    # Single inference
    result = engine.infer(
        "What are the key differences between FP16 and INT8 quantization?",
        max_tokens=128,
        temperature=0.7,
    )
    print(result.text)
    print(f"\n{result.tokens_per_sec:.1f} tokens/sec | "
          f"{result.latency_ms:.0f} ms | "
          f"{result.tokens_generated} tokens")

    # Batch inference
    prompts = [
        "Explain KV cache in one paragraph.",
        "What is flash attention?",
        "How does GGUF quantization preserve model quality?",
    ]
    results = engine.batch_generate(prompts, max_tokens=96)
    for i, r in enumerate(results):
        print(f"\n--- Result {i + 1} ({r.tokens_per_sec:.1f} tok/s) ---")
        print(r.text)

    engine.unload_model()
```

### Cell 5: Review metrics

```python
print("Inference complete. Review metrics in your OTLP backend if telemetry was enabled.")
```

---

## Recommended notebooks

For deeper Kaggle workflows, explore these production-tested notebooks:

- [01 Quickstart](../notebooks/01-quickstart-llamatelemetry-v0-1-1-e1.md) --
  Foundation setup and first inference.
- [03 Multi-GPU Inference](../notebooks/03-multi-gpu-inference-llamatelemetry-v0-1-1-e1.md) --
  Dual-T4 layer splitting and multi-GPU configurations.
- [06 Split-GPU Graphistry](../notebooks/06-split-gpu-graphistry-llamatelemetry-v0-1-1-e1.md) --
  LLM on GPU 0, graph visualization on GPU 1.
- [09 Large Models Kaggle](../notebooks/09-large-models-kaggle-llamatelemetry-e3.md) --
  Running 70B models on dual T4 with aggressive quantization.
- [14 OpenTelemetry Observability](../notebooks/14-opentelemetry-llm-observability-e5.md) --
  Full OTLP pipeline on Kaggle.
- [17 W&B Integration](../notebooks/17-llamatelemetry-wandb-kaggle-notebook-e2.md) --
  Weights & Biases experiment tracking.

---

## Next steps

- [Quickstart](quickstart.md) -- general-purpose quickstart for local
  workstations.
- [Kaggle Environment Guide](../guides/kaggle-environment.md) -- advanced
  Kaggle patterns, secrets management, and preset customization.
- [Telemetry and Observability](../guides/telemetry-observability.md) --
  OpenTelemetry configuration, Gen AI attributes, and dashboards.
- [Graphistry and RAPIDS](../guides/graphistry-rapids.md) -- GPU-accelerated
  graph visualization on the second T4.
