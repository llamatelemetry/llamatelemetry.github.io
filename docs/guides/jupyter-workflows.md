# Jupyter Workflows Guide

`llamatelemetry.jupyter` provides notebook-oriented helpers for interactive workflows.

## Main helpers

- `stream_generate(...)`
- `progress_generate(...)`
- `display_metrics(engine, as_dataframe=True)`
- `compare_temperatures(...)`
- `visualize_tokens(...)`
- `ChatWidget`

## Streaming generation in notebook

```python
from llamatelemetry.jupyter import stream_generate

stream_generate(engine, prompt="Explain KV cache simply.")
```

## Compare generation settings

```python
from llamatelemetry.jupyter import compare_temperatures

compare_temperatures(
    engine=engine,
    prompt="Write a short summary about GGUF.",
    temperatures=[0.2, 0.7, 1.0],
)
```

## Display runtime metrics

```python
from llamatelemetry.jupyter import display_metrics

display_metrics(engine, as_dataframe=True)
```

## Widget-based interaction

`ChatWidget` can provide a notebook-native chat UI if notebook dependencies are installed.

## Recommended notebook setup

1. Create and test `InferenceEngine`.
2. Run a short warmup prompt.
3. Use `progress_generate` or `stream_generate`.
4. Capture telemetry/metrics snapshots after each experiment.
