# Client API Reference (`llamatelemetry.api.client`)

## Core class

- `LlamaCppClient(base_url="http://127.0.0.1:8080", api_key=None, timeout=600.0, verify_ssl=True)`

## Sub-APIs

- `client.chat` -> `ChatCompletionsAPI`
- `client.embeddings` -> `EmbeddingsClientAPI`
- `client.models` -> `ModelsClientAPI`
- `client.slots` -> `SlotsClientAPI`
- `client.lora` -> `LoraClientAPI`

## Main methods

- Health and readiness:
  - `health()`
  - `is_ready()`
  - `wait_until_ready(timeout=60.0, poll_interval=1.0)`
- Server properties:
  - `props()`
  - `set_props(**kwargs)`
- Native generation:
  - `complete(...)`
  - `infill(...)`
- Text processing:
  - `tokenize(...)`
  - `detokenize(tokens)`
  - `apply_template(messages)`
- Embedding/ranking:
  - `embed(content, embd_normalize=2)`
  - `rerank(query, documents, top_n=None)`
- Metrics:
  - `metrics()`

## Dataclasses

- `Message`
- `Choice`
- `Usage`
- `Timings`
- `CompletionResponse`
- `EmbeddingData`
- `EmbeddingsResponse`
- `RerankResult`
- `RerankResponse`
- `TokenizeResponse`
- `ModelInfo`
- `SlotInfo`
- `HealthStatus`
- `LoraAdapter`

## Example

```python
from llamatelemetry.api import LlamaCppClient

client = LlamaCppClient("http://127.0.0.1:8080")
resp = client.chat.completions.create(
    messages=[{"role": "user", "content": "Summarize KV cache in one paragraph."}],
    max_tokens=120,
)
print(resp.choices[0].message.content)
```
