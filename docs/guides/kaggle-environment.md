---
title: Kaggle Environment Guide
description: Reference guide to llamatelemetry's Kaggle-focused helpers, including KaggleEnvironment, presets, split_gpu_session, OTLP secret loading, and telemetry client setup.
---

# Kaggle Environment

The `llamatelemetry.kaggle` package is one of the clearest themes in the
uploaded SDK snapshot. It packages together environment detection, presets,
secrets helpers, split-GPU context management, and a few notebook-friendly
pipeline helpers.

## What the Kaggle package exports

The public `llamatelemetry.kaggle` namespace currently exports these important
surfaces:

- `KaggleEnvironment`
- `quick_setup`
- `ServerPreset`, `TensorSplitMode`, `PresetConfig`, `get_preset_config`
- `KaggleSecrets`, `auto_load_secrets`
- `GPUContext`, `rapids_gpu`, `llm_gpu`, `single_gpu`, `split_gpu_session`
- `KagglePipelineConfig`
- `load_grafana_otlp_env_from_kaggle`
- `start_server_from_preset`
- `setup_otel_and_client`

## 1. Inspect the runtime with `KaggleEnvironment`

```python
from llamatelemetry.kaggle import KaggleEnvironment

env = KaggleEnvironment()

print("gpu_count:", env.gpu_count)
print("gpu_names:", env.gpu_names)
print("vram_per_gpu_gb:", env.vram_per_gpu_gb)
print("total_vram_gb:", env.total_vram_gb)
print("compute_capability:", env.compute_capability)
print("preset:", env.preset)
```

The uploaded package exposes real environment fields here, so this is a better
docs entry point than presenting Kaggle as a vague concept page.

## 2. Presets

`ServerPreset` gives you named configurations for common environments.

```python
from llamatelemetry.kaggle import ServerPreset, get_preset_config

cfg = get_preset_config(ServerPreset.KAGGLE_DUAL_T4)
print(cfg)
```

In the current snapshot, the enum members include:

- `AUTO`
- `KAGGLE_DUAL_T4`
- `KAGGLE_SINGLE_T4`
- `COLAB_T4`
- `COLAB_A100`
- `LOCAL_3090`
- `LOCAL_4090`
- `CPU_ONLY`

### Converting a preset to kwargs

```python
cfg = get_preset_config(ServerPreset.KAGGLE_DUAL_T4)

print(cfg.to_server_kwargs())
print(cfg.to_load_kwargs())
```

This is useful because it keeps your notebook logic aligned with the package’s
own tuning choices instead of scattering manual values everywhere.

## 3. Start a server from a preset

The actual helper signature is:

- `start_server_from_preset(model_path, preset, extra_args=None)`

Example:

```python
from llamatelemetry.models import load_model_smart
from llamatelemetry.kaggle import start_server_from_preset, ServerPreset

model_path = load_model_smart("gemma-3-1b-Q4_K_M", interactive=False)
manager = start_server_from_preset(str(model_path), ServerPreset.KAGGLE_DUAL_T4)

print(manager.server_url)
print(manager.check_server_health())
```

That example matches the current code more closely than older docs that passed
fields the helper does not currently accept.

## 4. Split GPU contexts

The current package exports `split_gpu_session(llm_gpu=0, graph_gpu=1)`.

```python
from llamatelemetry.kaggle import split_gpu_session

with split_gpu_session(llm_gpu=0, graph_gpu=1):
    print("LLM GPU and analytics GPU are split")
```

This is the core pattern for notebooks that want to keep one GPU focused on
inference while leaving another available for Graphistry, RAPIDS, or adjacent
CUDA workloads.

You can also use the more specific helpers:

```python
from llamatelemetry.kaggle import rapids_gpu, llm_gpu
```

## 5. Secrets handling

```python
from llamatelemetry.kaggle import KaggleSecrets

secrets = KaggleSecrets()
print(secrets.get("HF_TOKEN"))
```

The package also exports convenience helpers such as:

- `auto_load_secrets()`
- `setup_huggingface_auth()`
- `setup_graphistry_auth()`

These are best documented as notebook helpers, not as mandatory project setup.

## 6. OTLP environment loading for Kaggle

```python
from llamatelemetry.kaggle import load_grafana_otlp_env_from_kaggle

print(load_grafana_otlp_env_from_kaggle())
```

This helper loads Grafana-oriented OTLP secrets into `OTEL_*` environment
variables so other telemetry code can reuse them.

## 7. Telemetry bundle setup

`KagglePipelineConfig` is lightweight in the current package. It is focused on
telemetry-related configuration rather than being a giant all-in-one orchestration
object.

```python
from llamatelemetry.kaggle import KagglePipelineConfig, setup_otel_and_client

cfg = KagglePipelineConfig(
    service_name="llamatelemetry-kaggle",
    service_version="0.1.1",
    enable_graphistry=False,
    enable_llama_metrics=True,
    llama_metrics_interval=5.0,
)

bundle = setup_otel_and_client("http://127.0.0.1:8080", cfg)
print(bundle.keys())
```

The return value is a dictionary that includes the tracer, meter,
instrumented client, and GPU metrics collector.

## 8. `quick_setup()` and `KaggleEnvironment.setup()`

The package also documents a one-liner setup story through the Kaggle package.
That is fine to mention, but the docs should keep it framed as a convenience
entry point and not the only correct way to use the SDK.

```python
from llamatelemetry.kaggle import quick_setup

env = quick_setup()
print(env)
```

## 9. Recommended documentation stance

This page now uses a stricter wording model:

- **present in package** means there is real code and public export
- **recommended** means it fits the SDK’s strongest intended path
- **validated** should be reserved for notebooks, releases, or demos you have
  actually exercised

That distinction matters a lot for Kaggle-related docs because the project
contains both solid helper surfaces and more aspirational ecosystem integrations.

## Related pages

- [Kaggle Quickstart](../get-started/kaggle-quickstart.md)
- [Telemetry and Observability](telemetry-observability.md)
- [Quickstart](../get-started/quickstart.md)
