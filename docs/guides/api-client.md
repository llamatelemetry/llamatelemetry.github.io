# API Client Guide

`llamatelemetry.api.LlamaCppClient` is a typed HTTP client for llama.cpp server endpoints.

## Client initialization

```python
from llamatelemetry.api import LlamaCppClient

client = LlamaCppClient(
    base_url="http://127.0.0.1:8080",
    api_key=None,
    timeout=600.0,
)
```

## OpenAI-compatible chat

```python
resp = client.chat.completions.create(
    messages=[{"role": "user", "content": "What is quantization?"}],
    max_tokens=120,
    temperature=0.7,
)
print(resp.choices[0].message.content)
```

## Native completion endpoint

```python
resp = client.complete(
    prompt="The key tradeoff in quantization is",
    n_predict=100,
    temperature=0.7,
    top_k=40,
    top_p=0.95,
)
print(resp.choices[0].text)
```

## Embeddings

```python
emb = client.embeddings.create(input=["hello", "world"])
print(len(emb.data), len(emb.data[0].embedding))
```

## Tokenization and templates

```python
tok = client.tokenize("Hello llama.cpp")
text = client.detokenize(tok.tokens)
prompt = client.apply_template([{"role": "user", "content": "Hi"}])
```

## Health and readiness

```python
health = client.health()
ready = client.wait_until_ready(timeout=30)
```

## Slots and LoRA

```python
slots = client.slots.list()
adapters = client.lora.list()
```

## Metrics endpoint

```python
metrics_text = client.metrics()
print(metrics_text[:400])
```

## When to use this API instead of `InferenceEngine`

Use `LlamaCppClient` when you need:

- Direct access to endpoint-level behaviors
- Full control over native sampling parameters
- OpenAI-compatible and native endpoint interop
- Slot/LoRA/model management operations
