# Kaggle Environment

llamatelemetry provides first-class support for Kaggle notebooks with GPU accelerators. The `kaggle` module handles environment detection, preset configurations for T4 GPUs, secrets management, GPU context splitting, and end-to-end pipeline orchestration.

## Overview

The Kaggle integration includes:

- **KaggleEnvironment** -- detects runtime capabilities (GPU count, VRAM, accelerator type)
- **ServerPreset** -- pre-tested configurations for common Kaggle GPU setups
- **PresetConfig** -- converts presets into `ServerManager` and `load_model()` kwargs
- **split_gpu_session()** -- context manager for dedicating GPUs to different tasks
- **KaggleSecrets** -- retrieves notebook secrets for API keys and OTLP credentials
- **Pipeline helpers** -- `load_grafana_otlp_env_from_kaggle()`, `start_server_from_preset()`, `setup_otel_and_client()`

## Server Presets

Presets provide optimized configurations that have been tested on specific Kaggle accelerator tiers:

```python
from llamatelemetry.kaggle.presets import get_preset_config, ServerPreset

# Get configuration for Kaggle dual T4 setup
preset = get_preset_config(ServerPreset.KAGGLE_DUAL_T4)
print(preset)
```

### Available Presets

| Preset | GPUs | VRAM | Description |
|--------|------|------|-------------|
| `ServerPreset.KAGGLE_DUAL_T4` | 2x Tesla T4 | 2x 16 GB | Kaggle GPU T4 x2 accelerator |
| `ServerPreset.KAGGLE_SINGLE_T4` | 1x Tesla T4 | 16 GB | Kaggle GPU T4 x1 accelerator |
| `ServerPreset.COLAB_T4` | 1x Tesla T4 | 16 GB | Google Colab free-tier GPU |
| `ServerPreset.LOCAL_RTX_3090` | 1x RTX 3090 | 24 GB | Local development |
| `ServerPreset.LOCAL_RTX_4090` | 1x RTX 4090 | 24 GB | Local development |

### PresetConfig

Each preset returns a `PresetConfig` dataclass with methods to generate kwargs:

```python
preset = get_preset_config(ServerPreset.KAGGLE_DUAL_T4)

# Convert to ServerManager.start_server() kwargs
server_kwargs = preset.to_server_kwargs()
print(server_kwargs)
# {'gpu_layers': 99, 'ctx_size': 4096, 'n_parallel': 2,
#  'batch_size': 512, 'ubatch_size': 128, ...}

# Convert to InferenceEngine.load_model() kwargs
load_kwargs = preset.to_load_kwargs()
print(load_kwargs)
# {'gpu_layers': 99, 'ctx_size': 4096, 'n_parallel': 2, ...}
```

## KaggleEnvironment

Detect the current Kaggle runtime:

```python
from llamatelemetry.kaggle.environment import KaggleEnvironment

env = KaggleEnvironment()
print(f"Is Kaggle: {env.is_kaggle}")
print(f"GPU count: {env.gpu_count}")
print(f"GPU model: {env.gpu_model}")
print(f"VRAM per GPU: {env.vram_per_gpu_gb} GB")
print(f"Total VRAM: {env.total_vram_gb} GB")
print(f"Accelerator: {env.accelerator}")
```

## Split GPU Session

On dual-GPU Kaggle instances, dedicate one GPU to LLM inference and the other to graph analytics or visualization:

```python
from llamatelemetry.kaggle.gpu_context import split_gpu_session

# GPU 0 for inference, GPU 1 for Graphistry/RAPIDS
with split_gpu_session(llm_gpu=0, graph_gpu=1):
    # Inside this context:
    # - CUDA_VISIBLE_DEVICES is set for the LLM server
    # - The graph GPU is reserved for analytics workloads

    engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)
    result = engine.infer("What is CUDA?", max_tokens=128)
    print(result.text)
```

### How It Works

`split_gpu_session()` sets `CUDA_VISIBLE_DEVICES` to control which GPU the llama-server process uses. The other GPU remains available for PyTorch, RAPIDS cuGraph, or Graphistry workloads.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_gpu` | `int` | `0` | GPU index for LLM inference |
| `graph_gpu` | `int` | `1` | GPU index for analytics |

## KaggleSecrets

Retrieve secrets stored in the Kaggle notebook settings:

```python
from llamatelemetry.kaggle.secrets import KaggleSecrets

secrets = KaggleSecrets()

# Get individual secrets
otlp_endpoint = secrets.get("GRAFANA_OTLP_ENDPOINT")
otlp_token = secrets.get("GRAFANA_OTLP_TOKEN")
graphistry_user = secrets.get("GRAPHISTRY_USERNAME")
graphistry_pass = secrets.get("GRAPHISTRY_PASSWORD")
hf_token = secrets.get("HF_TOKEN")
```

### Required Secrets for Full Pipeline

| Secret Name | Used By | Description |
|-------------|---------|-------------|
| `GRAFANA_OTLP_ENDPOINT` | Telemetry | Grafana OTLP gateway URL |
| `GRAFANA_OTLP_TOKEN` | Telemetry | Base64 `instanceId:token` |
| `GRAPHISTRY_USERNAME` | Graphistry | Graphistry Hub username |
| `GRAPHISTRY_PASSWORD` | Graphistry | Graphistry Hub password |
| `GRAPHISTRY_SERVER` | Graphistry | Graphistry server URL |
| `HF_TOKEN` | Model downloads | HuggingFace access token |

## Pipeline Helpers

### load_grafana_otlp_env_from_kaggle()

Loads OTLP credentials from Kaggle secrets into environment variables:

```python
from llamatelemetry.kaggle.pipeline import load_grafana_otlp_env_from_kaggle

# Sets OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS
load_grafana_otlp_env_from_kaggle()
```

### start_server_from_preset()

Starts a llama-server using a preset configuration:

```python
from llamatelemetry.kaggle.pipeline import start_server_from_preset
from llamatelemetry.kaggle.presets import ServerPreset

manager = start_server_from_preset(
    preset=ServerPreset.KAGGLE_DUAL_T4,
    model_path="/kaggle/working/model.gguf",
)
# Server is now running and ready
```

### setup_otel_and_client()

Combines telemetry setup and instrumented client creation:

```python
from llamatelemetry.kaggle.pipeline import setup_otel_and_client

tracer, meter, client = setup_otel_and_client(
    service_name="kaggle-demo",
    server_url="http://127.0.0.1:8090",
)

# Client is already instrumented with telemetry
response = client.chat_completions({
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64,
})
```

## KagglePipelineConfig

For complex pipelines, use the configuration dataclass:

```python
from llamatelemetry.kaggle.pipeline import KagglePipelineConfig

config = KagglePipelineConfig(
    model_name="gemma-3-1b-Q4_K_M",
    preset=ServerPreset.KAGGLE_DUAL_T4,
    enable_telemetry=True,
    enable_graphistry=True,
    llm_gpu=0,
    graph_gpu=1,
)
```

## Recommended Models for Kaggle T4

### Single T4 (16 GB VRAM)

| Model | Size | Context | Use Case |
|-------|------|---------|----------|
| `gemma-3-1b-Q4_K_M` | ~0.8 GB | 8192 | Fast prototyping |
| `llama-3.2-1b-Q4_K_M` | ~0.9 GB | 4096 | General purpose |
| `phi-4-mini-Q4_K_M` | ~2.3 GB | 4096 | Coding, reasoning |
| `qwen-2.5-1.5b-Q4_K_M` | ~1.2 GB | 4096 | Multilingual |

### Dual T4 (2x 16 GB VRAM)

With layer splitting across two GPUs, larger models become feasible:

| Model | Size | Split | Use Case |
|-------|------|-------|----------|
| `llama-3.1-8b-Q4_K_M` | ~4.9 GB | 50/50 | High-quality general |
| `mistral-7b-Q4_K_M` | ~4.4 GB | 50/50 | Instruction following |
| `gemma-2-9b-Q4_K_M` | ~5.4 GB | 50/50 | Advanced reasoning |

## Complete Kaggle Notebook Example

```python
# Cell 1: Install
# !pip install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0

# Cell 2: Setup
import llamatelemetry as lt
from llamatelemetry.kaggle.presets import get_preset_config, ServerPreset
from llamatelemetry.kaggle.pipeline import (
    load_grafana_otlp_env_from_kaggle,
    setup_otel_and_client,
)
from llamatelemetry.kaggle.gpu_context import split_gpu_session
from llamatelemetry.kaggle.environment import KaggleEnvironment

# Cell 3: Detect environment
env = KaggleEnvironment()
print(f"GPUs: {env.gpu_count}x {env.gpu_model}")
print(f"Total VRAM: {env.total_vram_gb} GB")

# Cell 4: Load OTLP credentials
load_grafana_otlp_env_from_kaggle()

# Cell 5: Start inference with split GPU
with split_gpu_session(llm_gpu=0, graph_gpu=1):
    # Create engine with telemetry
    engine = lt.InferenceEngine(
        server_url="http://127.0.0.1:8090",
        enable_telemetry=True,
        telemetry_config={
            "service_name": "kaggle-llm-demo",
        },
    )

    # Load model using preset settings
    preset = get_preset_config(ServerPreset.KAGGLE_DUAL_T4)
    engine.load_model(
        "gemma-3-1b-Q4_K_M",
        auto_start=True,
        **preset.to_load_kwargs(),
    )

    # Run inference
    result = engine.infer(
        "Explain GPU memory hierarchy in 3 sentences.",
        max_tokens=128,
        temperature=0.7,
    )
    print(result.text)
    print(f"Speed: {result.tokens_per_sec:.1f} tok/s")

    # Batch inference
    prompts = [
        "What is CUDA?",
        "What is tensor parallelism?",
        "What is flash attention?",
    ]
    results = engine.batch_infer(prompts, max_tokens=64)
    for r in results:
        print(f"[{r.tokens_per_sec:.0f} tok/s] {r.text[:80]}...")

    engine.unload_model()
```

## Troubleshooting Kaggle Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `gpu_count=0` | CPU-only runtime | Enable GPU accelerator in notebook settings |
| Secret not found | Secret not added | Add secrets in Kaggle notebook settings panel |
| OOM on model load | Model too large | Use a smaller model or lower `ctx_size` |
| Slow first cell | Bootstrap downloading | Expected on first run (~961 MB download) |
| Server timeout | Model loading slowly | Increase `wait_ready()` timeout |

## Best Practices

- **Always check `KaggleEnvironment`** at the start of your notebook to verify GPU availability.
- **Use presets** instead of manually configuring GPU layers and batch sizes.
- **Split GPUs** when combining inference with Graphistry visualization.
- **Store credentials as Kaggle secrets** -- never hardcode API keys in notebooks.
- **Use small models** (1-3B parameters) for interactive notebook workflows.
- **Install once** in the first cell, not in every cell.

## Related

- [Inference Engine](inference-engine.md) -- high-level inference API
- [Server Management](server-management.md) -- server configuration details
- [Telemetry and Observability](telemetry-observability.md) -- OTLP export details
- [Graphistry and RAPIDS](graphistry-rapids.md) -- visualization on the second GPU
- [Kaggle API Reference](../reference/kaggle-api.md)
