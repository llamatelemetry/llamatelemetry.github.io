# Client API Reference

The `LlamaCppClient` provides a comprehensive, type-safe Python client for all llama.cpp server endpoints. It supports both OpenAI-compatible APIs and native llama.cpp endpoints, with structured response dataclasses and optional SSE streaming.

**Module:** `llamatelemetry.api.client`

---

## LlamaCppClient

### Constructor

```python
class LlamaCppClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        api_key: Optional[str] = None,
        timeout: float = 600.0,
        verify_ssl: bool = True,
    )
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `str` | `"http://127.0.0.1:8080"` | Server base URL |
| `api_key` | `Optional[str]` | `None` | API key for Bearer authentication |
| `timeout` | `float` | `600.0` | Request timeout in seconds |
| `verify_ssl` | `bool` | `True` | Verify SSL certificates |

### Sub-API Properties

| Property | Type | Description |
|----------|------|-------------|
| `client.chat` | `ChatCompletionsAPI` | OpenAI-compatible chat completions (`/v1/chat/completions`) |
| `client.embeddings` | `EmbeddingsClientAPI` | Embeddings API (`/v1/embeddings`) |
| `client.models` | `ModelsClientAPI` | Model management (`/v1/models`) |
| `client.slots` | `SlotsClientAPI` | Slot management (`/slots`) |
| `client.lora` | `LoraClientAPI` | LoRA adapter management (`/lora-adapters`) |

```python
from llamatelemetry.api.client import LlamaCppClient

client = LlamaCppClient("http://localhost:8080")
```

---

## Health and Server Endpoints

### health

```python
def health(self) -> HealthStatus
```

Check server health. Returns a `HealthStatus` with `status` (`"ok"`, `"loading"`, or error), `slots_idle`, and `slots_processing`.

### is_ready

```python
def is_ready(self) -> bool
```

Returns `True` if the server status is `"ok"`.

### wait_until_ready

```python
def wait_until_ready(self, timeout: float = 60.0, poll_interval: float = 1.0) -> bool
```

Block until the server is ready or timeout. Returns `True` if the server became ready.

### props

```python
def props(self) -> Dict[str, Any]
```

Get server global properties from `/props`. Returns dictionary with `default_generation_settings`, `total_slots`, `model_path`, `chat_template`, `modalities`, `is_sleeping`.

### set_props

```python
def set_props(self, **kwargs) -> Dict[str, Any]
```

Set server global properties (requires `--props` server flag).

### metrics

```python
def metrics(self) -> str
```

Get Prometheus-compatible metrics text from `/metrics`. Requires the `--metrics` server flag.

---

## Chat Completion

### chat_completion (convenience)

```python
def chat_completion(
    self,
    messages: List[Dict[str, Any]],
    **kwargs,
) -> Union[CompletionResponse, Iterator[Dict[str, Any]]]
```

Convenience wrapper that delegates to `client.chat.completions.create()`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `List[Dict[str, Any]]` | *required* | Chat messages (OpenAI format) |
| `**kwargs` | | | All parameters supported by `ChatCompletionsAPI.create()` |

```python
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is CUDA?"},
    ],
    max_tokens=200,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

### chat.completions.create

```python
def create(
    self,
    messages: List[Dict[str, Any]],
    model: str = "gpt-codex-5.3",
    max_tokens: Optional[int] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    n: int = 1,
    stream: bool = False,
    stop: Optional[Union[str, List[str]]] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    logit_bias: Optional[Dict[str, float]] = None,
    user: Optional[str] = None,
    response_format: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    seed: Optional[int] = None,
    # llama.cpp-specific extensions
    mirostat: int = 0,
    mirostat_tau: float = 5.0,
    mirostat_eta: float = 0.1,
    grammar: Optional[str] = None,
    min_p: float = 0.05,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    **kwargs,
) -> Union[CompletionResponse, Iterator[Dict[str, Any]]]
```

Full OpenAI-compatible chat completion endpoint with llama.cpp extensions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `List[Dict]` | *required* | Chat messages with `role` and `content` |
| `model` | `str` | `"gpt-codex-5.3"` | Model identifier |
| `max_tokens` | `Optional[int]` | `None` | Maximum tokens to generate |
| `temperature` | `float` | `1.0` | Sampling temperature (0.0-2.0) |
| `top_p` | `float` | `1.0` | Nucleus sampling (1.0 = disabled) |
| `n` | `int` | `1` | Number of completions to generate |
| `stream` | `bool` | `False` | Enable SSE streaming |
| `stop` | `Optional[Union[str, List[str]]]` | `None` | Stop sequences |
| `presence_penalty` | `float` | `0.0` | Presence penalty |
| `frequency_penalty` | `float` | `0.0` | Frequency penalty |
| `logit_bias` | `Optional[Dict[str, float]]` | `None` | Token logit biases |
| `response_format` | `Optional[Dict]` | `None` | Response format (`json_object`, `json_schema`) |
| `tools` | `Optional[List[Dict]]` | `None` | Tool/function definitions for function calling |
| `tool_choice` | `Optional[Union[str, Dict]]` | `None` | Tool selection mode |
| `seed` | `Optional[int]` | `None` | RNG seed for reproducibility |
| `grammar` | `Optional[str]` | `None` | BNF grammar for constrained generation (llama.cpp) |
| `min_p` | `float` | `0.05` | Min-p sampling (llama.cpp) |
| `top_k` | `int` | `40` | Top-k sampling (llama.cpp) |
| `repeat_penalty` | `float` | `1.1` | Repeat penalty (llama.cpp) |
| `mirostat` | `int` | `0` | Mirostat mode: 0=off, 1=v1, 2=v2 (llama.cpp) |

**Returns:** `CompletionResponse` or streaming iterator when `stream=True`.

```python
# Standard chat
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
    temperature=0.7,
)
print(response.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
    max_tokens=500,
):
    print(chunk.get("choices", [{}])[0].get("delta", {}).get("content", ""), end="")

# Structured output with JSON schema
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "List 3 colors"}],
    response_format={"type": "json_schema", "json_schema": {
        "name": "colors", "schema": {"type": "object", "properties": {
            "colors": {"type": "array", "items": {"type": "string"}}
        }}
    }},
)
```

---

## Native Completion

### complete

```python
def complete(
    self,
    prompt: Union[str, List[Union[str, int]]],
    n_predict: int = -1,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    min_p: float = 0.05,
    repeat_penalty: float = 1.1,
    repeat_last_n: int = 64,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    mirostat: int = 0,
    mirostat_tau: float = 5.0,
    mirostat_eta: float = 0.1,
    grammar: Optional[str] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    seed: int = -1,
    stop: Optional[List[str]] = None,
    stream: bool = False,
    cache_prompt: bool = True,
    n_probs: int = 0,
    samplers: Optional[List[str]] = None,
    dry_multiplier: float = 0.0,
    dry_base: float = 1.75,
    dry_allowed_length: int = 2,
    dry_penalty_last_n: int = -1,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.1,
    dynatemp_range: float = 0.0,
    dynatemp_exponent: float = 1.0,
    typical_p: float = 1.0,
    id_slot: int = -1,
    return_tokens: bool = False,
    **kwargs,
) -> Union[CompletionResponse, Iterator[Dict[str, Any]]]
```

Native llama.cpp completion endpoint (`/completion`) with full access to all sampling parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `Union[str, List]` | *required* | Input prompt (string or token array) |
| `n_predict` | `int` | `-1` | Max tokens to generate (-1 = unlimited) |
| `temperature` | `float` | `0.8` | Sampling temperature |
| `top_k` | `int` | `40` | Top-k sampling (0 = disabled) |
| `top_p` | `float` | `0.95` | Nucleus sampling (1.0 = disabled) |
| `min_p` | `float` | `0.05` | Min-p sampling (0.0 = disabled) |
| `repeat_penalty` | `float` | `1.1` | Repetition penalty |
| `grammar` | `Optional[str]` | `None` | BNF grammar for constrained generation |
| `json_schema` | `Optional[Dict]` | `None` | JSON schema for structured output |
| `seed` | `int` | `-1` | RNG seed (-1 = random) |
| `stream` | `bool` | `False` | Enable streaming |
| `cache_prompt` | `bool` | `True` | Reuse KV cache from previous request |
| `n_probs` | `int` | `0` | Return top N token probabilities |
| `samplers` | `Optional[List[str]]` | `None` | Custom sampler order |
| `dry_multiplier` | `float` | `0.0` | DRY sampling multiplier (0 = disabled) |
| `xtc_probability` | `float` | `0.0` | XTC sampling probability (0 = disabled) |
| `dynatemp_range` | `float` | `0.0` | Dynamic temperature range (0 = disabled) |
| `typical_p` | `float` | `1.0` | Locally typical sampling (1.0 = disabled) |
| `id_slot` | `int` | `-1` | Specific slot ID (-1 = auto) |
| `return_tokens` | `bool` | `False` | Return raw token IDs |

```python
response = client.complete(
    prompt="The capital of France is",
    n_predict=50,
    temperature=0.7,
)
print(response.choices[0].text)
```

### simple_completion

```python
def simple_completion(self, prompt: str, **kwargs) -> Union[str, CompletionResponse, Iterator]
```

Convenience wrapper that returns the generated text string when possible, falling back to the full `CompletionResponse` object.

---

## Embeddings

### embeddings.create (OpenAI-compatible)

```python
def create(
    self,
    input: Union[str, List[str]],
    model: str = "text-embedding-ada-002",
    encoding_format: str = "float",
    dimensions: Optional[int] = None,
) -> EmbeddingsResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `Union[str, List[str]]` | *required* | Text(s) to embed |
| `model` | `str` | `"text-embedding-ada-002"` | Model identifier |
| `encoding_format` | `str` | `"float"` | Output format (`float` or `base64`) |
| `dimensions` | `Optional[int]` | `None` | Embedding dimensions |

**Returns:** `EmbeddingsResponse` with `data` (list of `EmbeddingData`), `model`, and `usage`.

### embed (native)

```python
def embed(
    self,
    content: Union[str, List[str]],
    embd_normalize: int = 2,
) -> List[List[float]]
```

Native embedding endpoint (`/embedding`). Returns raw embedding vectors. Normalization types: -1=none, 0=max absolute, 1=taxicab, 2=Euclidean (L2).

---

## Tokenization

### tokenize

```python
def tokenize(
    self,
    content: str,
    add_special: bool = False,
    parse_special: bool = True,
    with_pieces: bool = False,
) -> TokenizeResponse
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `str` | *required* | Text to tokenize |
| `add_special` | `bool` | `False` | Add BOS/EOS tokens |
| `parse_special` | `bool` | `True` | Parse special token syntax |
| `with_pieces` | `bool` | `False` | Return token string pieces alongside IDs |

**Returns:** `TokenizeResponse` with `tokens` list.

### detokenize

```python
def detokenize(self, tokens: List[int]) -> str
```

Convert token IDs back to text. Returns the decoded string.

```python
tokens = client.tokenize("Hello, world!")
print(tokens.tokens)  # [15496, 11, 995, 0]
text = client.detokenize(tokens.tokens)
print(text)  # "Hello, world!"
```

---

## Reranking

### rerank

```python
def rerank(
    self,
    query: str,
    documents: List[str],
    top_n: Optional[int] = None,
) -> RerankResponse
```

Rerank documents by relevance to a query. Requires a reranker model loaded with `--embedding --pooling rank` server flags.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | *required* | Query string |
| `documents` | `List[str]` | *required* | Documents to rank |
| `top_n` | `Optional[int]` | `None` | Return only top N results |

**Returns:** `RerankResponse` with `results` (list of `RerankResult` with `index`, `relevance_score`, `document`).

```python
results = client.rerank(
    query="What is a panda?",
    documents=["A panda is a bear", "Hello world", "Pandas eat bamboo"],
    top_n=2,
)
for r in results.results:
    print(f"  [{r.index}] score={r.relevance_score:.3f}")
```

---

## Code Infill

### infill

```python
def infill(
    self,
    input_prefix: str,
    input_suffix: str,
    input_extra: Optional[List[Dict[str, str]]] = None,
    prompt: Optional[str] = None,
    stream: bool = False,
    **kwargs,
) -> Union[CompletionResponse, Iterator[Dict[str, Any]]]
```

Fill-in-the-middle code completion. Requires a model that supports FIM tokens.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_prefix` | `str` | *required* | Code before the cursor |
| `input_suffix` | `str` | *required* | Code after the cursor |
| `input_extra` | `Optional[List[Dict]]` | `None` | Additional context files |
| `prompt` | `Optional[str]` | `None` | Text added after FIM_MID token |
| `stream` | `bool` | `False` | Enable streaming |

---

## Slot Management

### slots.list

```python
def list(self, fail_on_no_slot: bool = False) -> List[SlotInfo]
```

List server slots. Each `SlotInfo` has: `id`, `is_processing`, `n_ctx`, `n_predict`, `params`, `prompt`.

### slots.save / slots.restore / slots.erase

```python
def save(self, slot_id: int, filename: str) -> Dict[str, Any]
def restore(self, slot_id: int, filename: str) -> Dict[str, Any]
def erase(self, slot_id: int) -> Dict[str, Any]
```

Save, restore, or erase the KV cache for a specific slot.

---

## LoRA Adapter Management

### lora.list

```python
def list(self) -> List[LoraAdapter]
```

List loaded LoRA adapters. Each `LoraAdapter` has: `id`, `path`, `scale`.

### lora.set_scales

```python
def set_scales(self, adapters: List[Dict[str, Any]]) -> bool
```

Set LoRA adapter scales at runtime.

```python
client.lora.set_scales([
    {"id": 0, "scale": 0.5},
    {"id": 1, "scale": 0.8},
])
```

---

## Response Dataclasses

### CompletionResponse

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Response ID |
| `object` | `str` | Object type (`"text_completion"` or `"chat.completion"`) |
| `created` | `int` | Unix timestamp |
| `model` | `str` | Model identifier |
| `choices` | `List[Choice]` | List of completion choices |
| `usage` | `Optional[Usage]` | Token usage statistics |
| `timings` | `Optional[Timings]` | Performance timings |
| `system_fingerprint` | `Optional[str]` | System fingerprint |

### Choice

| Field | Type | Description |
|-------|------|-------------|
| `index` | `int` | Choice index |
| `message` | `Optional[Message]` | Chat message (for chat completions) |
| `text` | `Optional[str]` | Generated text (for native completions) |
| `finish_reason` | `Optional[str]` | Stop reason (`"stop"`, `"length"`, etc.) |

### Message

| Field | Type | Description |
|-------|------|-------------|
| `role` | `str` | Message role (`"assistant"`, `"user"`, `"system"`) |
| `content` | `str` | Message content |
| `tool_calls` | `Optional[List[Dict]]` | Tool call results |
| `reasoning_content` | `Optional[str]` | Reasoning content (if supported) |

### Usage

| Field | Type | Description |
|-------|------|-------------|
| `prompt_tokens` | `int` | Input token count |
| `completion_tokens` | `int` | Output token count |
| `total_tokens` | `int` | Total token count |

### Timings

| Field | Type | Description |
|-------|------|-------------|
| `prompt_n` | `int` | Tokens evaluated |
| `prompt_ms` | `float` | Prompt evaluation time (ms) |
| `prompt_per_token_ms` | `float` | Time per prompt token (ms) |
| `prompt_per_second` | `float` | Prompt tokens per second |
| `predicted_n` | `int` | Tokens predicted |
| `predicted_ms` | `float` | Prediction time (ms) |
| `predicted_per_token_ms` | `float` | Time per predicted token (ms) |
| `predicted_per_second` | `float` | Prediction tokens per second |
| `cache_n` | `int` | Cached tokens reused |

### Other Dataclasses

| Class | Fields | Description |
|-------|--------|-------------|
| `EmbeddingsResponse` | `object`, `data`, `model`, `usage` | Embeddings API response |
| `EmbeddingData` | `index`, `embedding`, `object` | Single embedding vector |
| `RerankResponse` | `model`, `results`, `usage` | Rerank API response |
| `RerankResult` | `index`, `relevance_score`, `document` | Single rerank result |
| `TokenizeResponse` | `tokens` | Tokenization result |
| `HealthStatus` | `status`, `slots_idle`, `slots_processing` | Health check response |
| `SlotInfo` | `id`, `is_processing`, `n_ctx`, `n_predict`, `params`, `prompt` | Slot status |
| `LoraAdapter` | `id`, `path`, `scale` | LoRA adapter info |
| `StopType` | Enum: `NONE`, `EOS`, `LIMIT`, `WORD` | Completion stop types |

---

## Related Modules

- [Core API](core-api.md) -- High-level InferenceEngine wraps this client
- [Telemetry API](telemetry-api.md) -- InstrumentedLlamaCppClient for auto-tracing
- [Server and Models](server-models.md) -- Server lifecycle management
