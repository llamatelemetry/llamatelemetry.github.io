# API Client

`llamatelemetry.api.LlamaCppClient` is an OpenAI-compatible HTTP client for `llama-server`. It exposes chat completions, embeddings, model listing, and low-level endpoints.

## Basic usage

```python
from llamatelemetry.api import LlamaCppClient

client = LlamaCppClient(base_url="http://127.0.0.1:8090")
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "What is GGUF?"}],
    max_tokens=64,
)
print(response.choices[0].message.content)
```

## Embeddings

```python
emb = client.embeddings.create(
    input=["llamatelemetry", "llama.cpp"],
    model="/path/to/model.gguf",
)
```

## Model listing

```python
models = client.models.list()
print(models)
```

## Slots and health

```python
print(client.slots.list())
print(client.health.check())
```

## Related reference

- [Client API](../reference/client-api.md)
- [Server and Models](../reference/server-models.md)
