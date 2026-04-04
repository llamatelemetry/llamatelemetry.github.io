---
title: Kaggle Quickstart
description: Practical Kaggle-first quickstart for llamatelemetry on GPU notebooks, including install, environment detection, presets, model loading, and optional OTLP telemetry.
---

# Kaggle Quickstart

This is the most important runtime path for the current project.

The uploaded SDK snapshot clearly orients much of the package around **Kaggle
GPU workflows**, especially **dual Tesla T4** usage. This page therefore keeps a
Kaggle-first story and separates it from generic local-Linux documentation.

## 1. Start a GPU notebook

In Kaggle, open a notebook and enable a GPU accelerator from the notebook
settings.

The strongest documented path is a **dual T4** session, but the package also
contains presets for single-T4 and nearby environments.

## 2. Install the package

First cell:

```python
!pip -q install --no-cache-dir --force-reinstall \
  git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1
```

Optional add-ons for observability or notebook integrations:

```python
!pip -q install opentelemetry-exporter-otlp-proto-http
!pip -q install pygraphistry pandas
!pip -q install wandb
!pip -q install sseclient-py
```

## 3. Check what Kaggle gave you

```python
import llamatelemetry as lt
from llamatelemetry.kaggle import KaggleEnvironment

env = KaggleEnvironment()

print("gpu_count:", env.gpu_count)
print("gpu_names:", env.gpu_names)
print("vram_per_gpu_gb:", env.vram_per_gpu_gb)
print("total_vram_gb:", env.total_vram_gb)
print("compute_capability:", env.compute_capability)
print("preset:", env.preset)
print("cuda:", lt.detect_cuda())
```

This is a better fit for the real package than hard-coding one exact expected
output block.

## 4. Use a preset

The package includes Kaggle-aware presets.

```python
from llamatelemetry.kaggle import ServerPreset, get_preset_config

cfg = get_preset_config(ServerPreset.KAGGLE_DUAL_T4)
print(cfg)
print(cfg.to_server_kwargs())
print(cfg.to_load_kwargs())
```

Available preset names in the uploaded snapshot include:

- `ServerPreset.AUTO`
- `ServerPreset.KAGGLE_DUAL_T4`
- `ServerPreset.KAGGLE_SINGLE_T4`
- `ServerPreset.COLAB_T4`
- `ServerPreset.COLAB_A100`
- `ServerPreset.LOCAL_3090`
- `ServerPreset.LOCAL_4090`
- `ServerPreset.CPU_ONLY`

## 5. Download or resolve a model

A Kaggle-friendly pattern is to resolve the model path explicitly first:

```python
from llamatelemetry.models import load_model_smart

model_path = load_model_smart(
    "gemma-3-1b-Q4_K_M",
    interactive=False,
)

print(model_path)
```

## 6. Start the server from a preset

The actual `start_server_from_preset()` signature in the uploaded SDK is:

- `model_path`
- `preset`
- optional `extra_args`

So the clean example is:

```python
from llamatelemetry.kaggle import start_server_from_preset, ServerPreset

manager = start_server_from_preset(
    model_path=str(model_path),
    preset=ServerPreset.KAGGLE_DUAL_T4,
)

print(manager.server_url)
print(manager.check_server_health())
```

This is more accurate than older docs that passed `model_name=` or other fields
that do not match the current helper.

## 7. Run inference through `InferenceEngine`

```python
import llamatelemetry as lt

engine = lt.InferenceEngine(
    server_url="http://127.0.0.1:8080",
    enable_telemetry=False,
)

result = engine.infer(
    "Summarize why Kaggle is useful for small GGUF experiments.",
    max_tokens=128,
    temperature=0.7,
)

print(result.text)
print(result.latency_ms)
print(result.tokens_per_sec)
```

## 8. Split GPU usage when you need it

The package exposes `split_gpu_session(llm_gpu=0, graph_gpu=1)`.

```python
from llamatelemetry.kaggle import split_gpu_session

with split_gpu_session(llm_gpu=0, graph_gpu=1):
    # place LLM work on one GPU and reserve the other for analytics-oriented work
    print("split GPU context active")
```

This is a good pattern when you want inference plus Graphistry or other CUDA
analysis work in one notebook.

## 9. Optional OTLP setup from Kaggle secrets

If you store telemetry credentials in Kaggle secrets, the package can load them
into `OTEL_*` environment variables.

```python
from llamatelemetry.kaggle import load_grafana_otlp_env_from_kaggle

otlp_env = load_grafana_otlp_env_from_kaggle()
print(otlp_env)
```

After that, you can enable telemetry in an engine or set up an instrumented
client.

## 10. One-stop telemetry client setup

```python
from llamatelemetry.kaggle import KagglePipelineConfig, setup_otel_and_client

cfg = KagglePipelineConfig(
    service_name="llamatelemetry-kaggle",
    service_version="0.1.1",
    enable_llama_metrics=True,
    llama_metrics_interval=5.0,
)

bundle = setup_otel_and_client("http://127.0.0.1:8080", cfg)
print(bundle.keys())
```

The helper returns a dictionary containing the tracer, meter, client, and GPU
metrics collector.

## 11. A compact end-to-end notebook flow

```python
!pip -q install --no-cache-dir --force-reinstall \
  git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1

import llamatelemetry as lt
from llamatelemetry.models import load_model_smart
from llamatelemetry.kaggle import (
    ServerPreset,
    start_server_from_preset,
)

model_path = load_model_smart("gemma-3-1b-Q4_K_M", interactive=False)
manager = start_server_from_preset(str(model_path), ServerPreset.KAGGLE_DUAL_T4)

engine = lt.InferenceEngine(server_url=manager.server_url, enable_telemetry=False)
result = engine.infer("Explain dual-T4 inference in plain English.", max_tokens=96)
print(result.text)
print(engine.get_metrics())
```

## What this page now keeps explicit

- Kaggle is the primary documented runtime story
- dual-T4 is the most opinionated path in the current package
- preset helpers are documented using their **real current signatures**
- OTLP, Graphistry, and W&B are documented as optional layers on top, not as
  mandatory parts of the core install
