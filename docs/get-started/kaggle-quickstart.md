# Kaggle Quickstart

`llamatelemetry` is optimized for Kaggle dual Tesla T4 workflows.

## Install in notebook

```python
!pip install -q --no-cache-dir --force-reinstall \
  git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

## One-liner environment setup

```python
from llamatelemetry.kaggle import KaggleEnvironment

env = KaggleEnvironment.setup(
    enable_telemetry=True,
    enable_graphistry=False,
    auto_load_secrets=True,
)
```

## Create engine and infer

```python
engine = env.create_engine("gemma-3-4b-Q4_K_M")
result = engine.infer("Summarize dual-T4 tensor split strategy in 4 points.")
print(result.text)
```

## Manual model download (optional)

```python
model_path = env.download_model(
    repo_id="unsloth/gemma-3-1b-it-GGUF",
    filename="gemma-3-1b-it-Q4_K_M.gguf",
)
```

## GPU context isolation

```python
with env.rapids_context():
    import cudf
    df = cudf.DataFrame({"x": [1, 2, 3]})
```

## Next

- [Kaggle Environment guide](../guides/kaggle-environment.md)
- [Graphistry and RAPIDS](../guides/graphistry-rapids.md)
- [Observability notebook track](../notebooks/observability.md)
