# Jupyter, Chat, and Embeddings API Reference

## Module: `llamatelemetry.jupyter`

- `is_jupyter_available()`
- `check_dependencies(require_widgets=False)`
- `stream_generate(...)`
- `progress_generate(...)`
- `display_metrics(...)`
- `ChatWidget`
- `compare_temperatures(...)`
- `visualize_tokens(...)`

---

## Module: `llamatelemetry.chat`

- `Message`
- `ChatEngine`
- `ConversationManager`

---

## Module: `llamatelemetry.embeddings`

- `EmbeddingEngine`
- `SemanticSearch`
- `TextClustering`
- `cosine_similarity(...)`
- `euclidean_distance(...)`
- `dot_product_similarity(...)`

## Example

```python
from llamatelemetry.embeddings import EmbeddingEngine, cosine_similarity

engine = EmbeddingEngine(server_url="http://127.0.0.1:8080")
vecs = engine.embed(["llama.cpp", "OpenTelemetry"])
print(cosine_similarity(vecs[0], vecs[1]))
```
