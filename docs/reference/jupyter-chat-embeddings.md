# Jupyter, Chat, and Embeddings API

## Jupyter helpers

`llamatelemetry.jupyter`:

- `is_jupyter_available()`
- `check_dependencies()`
- `stream_generate()`
- `progress_generate()`
- `display_metrics()`
- `compare_temperatures()`
- `visualize_tokens()`
- `ChatWidget`

## Chat

`llamatelemetry.chat`:

- `Message` — message model
- `ChatEngine` — conversation state manager
- `ConversationManager` — manage multiple sessions

## Embeddings

`llamatelemetry.embeddings`:

- `EmbeddingEngine`
- `SemanticSearch`
- `TextClustering`
- similarity helpers: `cosine_similarity`, `euclidean_distance`, `dot_product_similarity`

## Related docs

- [Jupyter Workflows](../guides/jupyter-workflows.md)
- [Inference Engine](../guides/inference-engine.md)
