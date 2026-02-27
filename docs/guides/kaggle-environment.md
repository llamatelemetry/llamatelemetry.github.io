# Kaggle Environment Guide

`llamatelemetry.kaggle` is designed to reduce notebook boilerplate for GPU setup, secrets, presets, and engine creation.

## Primary entrypoint

```python
from llamatelemetry.kaggle import KaggleEnvironment

env = KaggleEnvironment.setup(
    enable_telemetry=True,
    enable_graphistry=False,
    auto_load_secrets=True,
    split_gpu_mode=True,
    verbose=True,
)
```

## What setup does

- Detects GPUs, VRAM, and CUDA context via `api.multigpu`
- Selects a server preset (`KAGGLE_DUAL_T4`, `KAGGLE_SINGLE_T4`, etc.)
- Loads secrets (`HF_TOKEN`, Graphistry keys) when available
- Prepares telemetry and optional Graphistry registration

## Create a tuned engine

```python
engine = env.create_engine("gemma-3-4b-Q4_K_M", auto_start=True)
```

## Presets and tensor split

Use:

- `ServerPreset`
- `TensorSplitMode`
- `get_preset_config(...)`

to standardize launch behavior across notebooks.

## GPU context utilities

```python
with env.rapids_context():
    # RAPIDS on designated GPU
    pass

with env.llm_context():
    # LLM operations on configured GPUs
    pass
```

## Secrets helpers

From `llamatelemetry.kaggle.secrets`:

- `auto_load_secrets`
- `setup_huggingface_auth`
- `setup_graphistry_auth`

## Recommended notebook pattern

1. `env = KaggleEnvironment.setup(...)`
2. `engine = env.create_engine(...)`
3. Use `rapids_context()` for graph analytics workload separation.
