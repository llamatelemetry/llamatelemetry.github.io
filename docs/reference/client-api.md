# Client API

`llamatelemetry.api.client` provides an OpenAI-compatible interface for `llama-server`.

## LlamaCppClient

**Constructor:**

```python
LlamaCppClient(base_url="http://127.0.0.1:8090")
```

**Key properties:**

- `chat.completions` — chat completions API
- `embeddings` — embeddings API
- `models` — model listing API
- `slots` — slot status API
- `lora` — LoRA adapter API

## Chat completions

```python
client = LlamaCppClient(base_url="http://127.0.0.1:8090")
resp = client.chat.completions.create(
    messages=[{"role": "user", "content": "What is GGUF?"}],
    max_tokens=64,
)
```

## Embeddings

```python
emb = client.embeddings.create(
    input=["hello", "world"],
    model="/path/to/model.gguf",
)
```

## Models and slots

```python
print(client.models.list())
print(client.slots.list())
```

## Data models

The client defines dataclasses for structured responses:

- `Message`, `Choice`, `Usage`, `Timings`
- `CompletionResponse`, `EmbeddingsResponse`, `RerankResponse`
- `ModelInfo`, `SlotInfo`, `HealthStatus`

## Related modules

- [Core API](core-api.md)
- [Server and Models](server-models.md)
