# API Client

`LlamaCppClient` is an OpenAI-compatible HTTP client for interacting with `llama-server`. It provides both the familiar OpenAI-style nested interface (`client.chat.completions.create()`) and convenience methods for direct access to llama.cpp-specific features like tokenization, reranking, and slot management.

## Overview

The client layer provides:

- **OpenAI-compatible chat completions** via `client.chat.completions.create()`
- **Convenience methods** like `client.chat_completion()` (singular) for simpler usage
- **Embeddings** via `client.embed()` and `client.embeddings.create()`
- **Tokenization** via `client.tokenize()` and `client.detokenize()`
- **Reranking** via `client.rerank()`
- **Server management** endpoints (health, metrics, slots, models, LoRA)

!!! warning "Port Mismatch"
    `LlamaCppClient` defaults to port **8090**, while `ServerManager` and `InferenceEngine` default to port **8090**. When using `LlamaCppClient` with a llamatelemetry-managed server, always specify the correct port:
    ```python
    client = LlamaCppClient(base_url="http://127.0.0.1:8090")
    ```

## Creating a Client

```python
from llamatelemetry.api import LlamaCppClient

# Default -- port 8090 (llama.cpp default)
client = LlamaCppClient()

# Matching llamatelemetry ServerManager port
client = LlamaCppClient(base_url="http://127.0.0.1:8090")
```

## Chat Completions

### OpenAI-Style API

The nested `client.chat.completions.create()` matches the OpenAI Python SDK interface:

```python
response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is GGUF format?"},
    ],
    max_tokens=128,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
)

print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.completion_tokens}")
```

### Convenience Method

The `chat_completion()` method (singular) is a simpler wrapper:

```python
response = client.chat_completion(
    messages=[{"role": "user", "content": "Explain tensor cores."}],
    max_tokens=64,
    temperature=0.5,
)
print(response.choices[0].message.content)
```

### Streaming Chat

```python
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Write a poem about GPUs."}],
    max_tokens=128,
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### Chat Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `list[dict]` | required | Conversation messages with `role` and `content` |
| `max_tokens` | `int` | `128` | Maximum tokens to generate |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `top_p` | `float` | `0.9` | Nucleus sampling threshold |
| `top_k` | `int` | `40` | Top-K sampling |
| `stream` | `bool` | `False` | Enable streaming response |
| `seed` | `int` | `0` | Random seed for reproducibility |
| `stop` | `list[str]` | `None` | Stop sequences |
| `frequency_penalty` | `float` | `0.0` | Frequency penalty |
| `presence_penalty` | `float` | `0.0` | Presence penalty |

## Embeddings

Generate embeddings for text inputs:

```python
# Simple embedding
result = client.embed("What is CUDA?")
print(f"Embedding dimension: {len(result)}")

# OpenAI-style embedding
response = client.embeddings.create(
    input=["llamatelemetry", "llama.cpp", "CUDA programming"],
    model="/path/to/model.gguf",
)
for item in response.data:
    print(f"Index {item.index}: {len(item.embedding)} dimensions")
```

!!! note
    Embedding requires a model that supports embedding extraction. Not all GGUF models produce useful embeddings.

## Tokenization

### Tokenize Text

Convert text to token IDs:

```python
tokens = client.tokenize("Hello, world!")
print(f"Token count: {len(tokens)}")
print(f"Token IDs: {tokens}")
```

### Detokenize Tokens

Convert token IDs back to text:

```python
text = client.detokenize([1, 22557, 29892, 3186, 29991])
print(f"Text: {text}")
```

Tokenization is useful for:

- Counting tokens before inference to stay within context limits
- Debugging prompt formatting issues
- Implementing custom token-level processing

## Reranking

Rerank documents by relevance to a query:

```python
query = "What is flash attention?"
documents = [
    "Flash attention is a memory-efficient attention algorithm.",
    "CUDA is a parallel computing platform by NVIDIA.",
    "Attention mechanisms are core to transformer models.",
    "Python is a popular programming language.",
]

results = client.rerank(query=query, documents=documents)
for result in results:
    print(f"Score: {result['score']:.4f} -- {documents[result['index']][:60]}")
```

## Model Management

### List Models

```python
models = client.models.list()
for model in models:
    print(f"Model: {model.get('id')}")
```

### Get Model Details

```python
# Via the models endpoint
models = client.models.list()
if models:
    print(f"Loaded model: {models[0].get('id')}")
```

## Slot Management

Slots represent concurrent inference contexts in the server:

```python
# List active slots
slots = client.slots.list()
for slot in slots:
    print(f"Slot {slot['id']}: state={slot['state']}")
    if slot.get('prompt'):
        print(f"  Prompt: {slot['prompt'][:50]}...")
```

## LoRA Adapter Management

Hot-swap LoRA adapters on the running server:

```python
# List loaded adapters
adapters = client.lora.list()
print(f"Active adapters: {adapters}")

# Apply a LoRA adapter (if supported by the server build)
# client.lora.apply("/path/to/adapter.gguf")
```

## Health and Metrics

### Health Check

```python
health = client.health.check()
print(f"Status: {health.get('status')}")
# Returns: {"status": "ok"} or {"status": "loading model"} etc.
```

### Prometheus Metrics

```python
metrics = client.metrics.get()
print(metrics)
# Returns Prometheus-format text with server metrics
```

## InstrumentedLlamaCppClient

For telemetry-enabled workflows, use `InstrumentedLlamaCppClient` which automatically creates OpenTelemetry spans:

```python
from llamatelemetry.telemetry import InstrumentedLlamaCppClient

client = InstrumentedLlamaCppClient(base_url="http://127.0.0.1:8090")

# Note: uses chat_completions (plural) -- different from LlamaCppClient
response = client.chat_completions({
    "messages": [{"role": "user", "content": "What is GGUF?"}],
    "max_tokens": 64,
})
```

!!! warning "API Difference"
    `InstrumentedLlamaCppClient.chat_completions()` (plural) takes a raw payload dict, while `LlamaCppClient.chat_completion()` (singular) takes keyword arguments. These are different classes with different interfaces.

## Error Handling

```python
from llamatelemetry.api import LlamaCppClient
import requests

client = LlamaCppClient(base_url="http://127.0.0.1:8090")

try:
    response = client.chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=64,
    )
    print(response.choices[0].message.content)
except requests.ConnectionError:
    print("Server not running. Start it with ServerManager or InferenceEngine.")
except requests.Timeout:
    print("Request timed out. Model may be loading or prompt may be too long.")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Comparison: Client Classes

| Feature | `LlamaCppClient` | `InstrumentedLlamaCppClient` |
|---------|-------------------|------------------------------|
| Default port | 8090 | 8090 (via telemetry setup) |
| Chat method | `chat_completion()` (singular) | `chat_completions()` (plural) |
| OpenAI-style | `chat.completions.create()` | Not available |
| Telemetry | No | Auto-creates OTel spans |
| Embeddings | `embed()`, `embeddings.create()` | Not available |
| Tokenize | `tokenize()`, `detokenize()` | Not available |
| Rerank | `rerank()` | Not available |

## Best Practices

- **Always specify the port** when using `LlamaCppClient` with a llamatelemetry-managed server (port 8090).
- **Use the OpenAI-style API** (`chat.completions.create()`) for code that may later switch to OpenAI or other providers.
- **Use `InstrumentedLlamaCppClient`** when you need automatic telemetry span creation.
- **Check health before sending requests** in production code to handle server restarts gracefully.
- **Use tokenize() to count tokens** before sending large prompts to avoid context overflow.

## Complete Example

```python
from llamatelemetry.api import LlamaCppClient

# Connect to a running llama-server (started by ServerManager on port 8090)
client = LlamaCppClient(base_url="http://127.0.0.1:8090")

# Check health
health = client.health.check()
print(f"Server status: {health.get('status')}")

# Chat completion (OpenAI-style)
response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a CUDA expert."},
        {"role": "user", "content": "What are warp divergence issues?"},
    ],
    max_tokens=128,
    temperature=0.7,
)
print(response.choices[0].message.content)

# Tokenize a prompt to check length
tokens = client.tokenize("What are warp divergence issues?")
print(f"Prompt tokens: {len(tokens)}")

# Get embeddings
embedding = client.embed("CUDA warp divergence")
print(f"Embedding dims: {len(embedding)}")

# Check slots
slots = client.slots.list()
print(f"Active slots: {len(slots)}")
```

## Related

- [Inference Engine](inference-engine.md) -- high-level wrapper that manages client internally
- [Server Management](server-management.md) -- starts the server that this client connects to
- [Telemetry and Observability](telemetry-observability.md) -- instrumented client details
- [Client API Reference](../reference/client-api.md)
