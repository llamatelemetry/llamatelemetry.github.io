# Kaggle Environment

`llamatelemetry.kaggle` contains helpers for zero-boilerplate Kaggle setup, GPU context inspection, and secret management.

## Key modules

- `environment`: detect Kaggle runtime capabilities
- `presets`: recommended configuration for T4 dual-GPU
- `gpu_context`: VRAM and device properties
- `secrets`: Kaggle secret retrieval
- `pipeline`: end-to-end workflow orchestration

## Presets

```python
from llamatelemetry.api import kaggle_t4_dual_config

cfg = kaggle_t4_dual_config()
print(cfg)
```

## GPU context

```python
from llamatelemetry.kaggle.gpu_context import get_gpu_context

ctx = get_gpu_context()
print(ctx)
```

## Secrets

```python
from llamatelemetry.kaggle.secrets import KaggleSecrets

secrets = KaggleSecrets()
print(secrets.get("OTLP_ENDPOINT"))
```

## Pipeline (high level)

```python
from llamatelemetry.kaggle.pipeline import KagglePipeline

pipeline = KagglePipeline()
pipeline.prepare_environment()
```

## Related reference

- [Kaggle API](../reference/kaggle-api.md)
- [Kaggle Quickstart](../get-started/kaggle-quickstart.md)
