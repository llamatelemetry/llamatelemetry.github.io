# Jupyter Workflows

The `llamatelemetry.jupyter` module provides notebook-friendly helpers for streaming outputs and interactive UI.

## Key helpers

- `stream_generate` — stream tokens in real time
- `progress_generate` — batch generation with progress bars
- `display_metrics` — render metrics for quick inspection
- `ChatWidget` — interactive chat UI

## Example

```python
from llamatelemetry.jupyter import stream_generate
from llamatelemetry import InferenceEngine

engine = InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)

stream_generate(engine, "Explain CUDA streams.", max_tokens=64)
```

## Related reference

- [Jupyter, Chat, and Embeddings API](../reference/jupyter-chat-embeddings.md)
