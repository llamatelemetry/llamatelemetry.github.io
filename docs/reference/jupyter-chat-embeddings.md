# Jupyter, Chat & Embeddings API Reference

`llamatelemetry.jupyter` provides JupyterLab-optimized features for interactive inference.
`llamatelemetry.chat` provides OpenAI-compatible chat completion with conversation management.
`llamatelemetry.embeddings` provides text embedding generation with caching, similarity search,
and clustering.

```python
from llamatelemetry.jupyter import (
    is_jupyter_available, check_dependencies,
    stream_generate, progress_generate,
    display_metrics, compare_temperatures, visualize_tokens,
    ChatWidget,
)
from llamatelemetry.chat import (
    Message, ChatEngine, ConversationManager,
)
from llamatelemetry.embeddings import (
    EmbeddingEngine, SemanticSearch, TextClustering,
    cosine_similarity, euclidean_distance, dot_product_similarity,
)
```

---

## Jupyter Helpers

### is_jupyter_available()

```python
def is_jupyter_available() -> bool
```

**Returns:** `True` if running in a Jupyter/IPython environment.

### check_dependencies()

```python
def check_dependencies(require_widgets: bool = False) -> bool
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `require_widgets` | `bool` | `False` | Whether to require `ipywidgets` |

**Returns:** `True` if Jupyter and optional dependencies are available.

---

### stream_generate()

```python
def stream_generate(
    engine,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    show_timing: bool = True,
    markdown: bool = True,
    **kwargs,
) -> str
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine` | `InferenceEngine` | -- | InferenceEngine instance |
| `prompt` | `str` | -- | Input prompt |
| `max_tokens` | `int` | `256` | Maximum tokens to generate |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `show_timing` | `bool` | `True` | Display timing information after generation |
| `markdown` | `bool` | `True` | Render output as markdown (vs. preformatted text) |

Streams text generation with real-time IPython display updates. Falls back to non-streaming `engine.infer()` outside Jupyter.

**Returns:** Complete generated text.

```python
from llamatelemetry.jupyter import stream_generate
text = stream_generate(engine, "Write a haiku about AI")
```

---

### progress_generate()

```python
def progress_generate(
    engine,
    prompts: List[str],
    max_tokens: int = 128,
    **kwargs,
) -> List[str]
```

Batch generation with `tqdm` progress bar. Falls back to print-based progress if `tqdm` is not installed.

**Returns:** List of generated texts (empty string for failed generations).

---

### display_metrics()

```python
def display_metrics(engine, as_dataframe: bool = True)
```

Displays performance metrics from `engine.get_metrics()` as a Pandas DataFrame or HTML table.

---

### compare_temperatures()

```python
def compare_temperatures(
    engine,
    prompt: str,
    temperatures: List[float] = [0.3, 0.7, 1.0, 1.5],
    max_tokens: int = 100,
) -> Dict[float, str]
```

Generates outputs at different temperature settings and displays them side-by-side.

**Returns:** Dictionary mapping temperature values to generated text.

```python
results = compare_temperatures(engine, "The future of AI is", temperatures=[0.1, 0.7, 1.5])
```

---

### visualize_tokens()

```python
def visualize_tokens(text: str, engine=None)
```

Visualizes token boundaries in text. If `engine` is provided, uses the `/tokenize` endpoint to get actual token boundaries and displays them as styled HTML spans with token count.

---

## ChatWidget

Interactive chat widget for JupyterLab with text input, send/clear buttons, conversation history,
and model parameter sliders for temperature and max tokens.

### ChatWidget(engine, system_prompt, max_tokens, temperature)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine` | `InferenceEngine` | -- | InferenceEngine instance |
| `system_prompt` | `Optional[str]` | `None` | System prompt prepended to conversations |
| `max_tokens` | `int` | `256` | Default max tokens (adjustable via slider) |
| `temperature` | `float` | `0.7` | Default temperature (adjustable via slider) |

Raises `ImportError` if `ipywidgets` is not installed.

### ChatWidget.display()

```python
def display(self) -> None
```

Renders the chat widget in the notebook output cell.

```python
from llamatelemetry.jupyter import ChatWidget
chat = ChatWidget(engine, system_prompt="You are a helpful assistant.")
chat.display()
```

---

## Message

Represents a single message in a conversation.

```python
class Message:
    role: str                   # "system", "user", or "assistant"
    content: str                # Message text
    name: Optional[str]         # Optional sender name
    timestamp: float            # Unix timestamp (auto-set)
```

### Message(role, content, name=None)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `role` | `str` | -- | Message role (`"system"`, `"user"`, `"assistant"`) |
| `content` | `str` | -- | Message content |
| `name` | `Optional[str]` | `None` | Optional sender name |

### Message.to_dict()

```python
def to_dict(self) -> Dict[str, str]
```

**Returns:** OpenAI-compatible message dict with `role`, `content`, and optionally `name`.

---

## ChatEngine

Manages chat conversations with history, context window handling, and OpenAI-compatible
chat completion support.

### ChatEngine(engine, system_prompt, max_history, max_tokens, temperature)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine` | `InferenceEngine` | -- | InferenceEngine instance |
| `system_prompt` | `Optional[str]` | `None` | System prompt (added as first message) |
| `max_history` | `int` | `20` | Maximum messages to keep (trims non-system messages) |
| `max_tokens` | `int` | `256` | Default max tokens |
| `temperature` | `float` | `0.7` | Default temperature |

### ChatEngine.add_message() / add_system_message() / add_user_message() / add_assistant_message()

```python
def add_message(self, role: str, content: str, name: Optional[str] = None) -> ChatEngine
def add_system_message(self, content: str) -> ChatEngine
def add_user_message(self, content: str) -> ChatEngine
def add_assistant_message(self, content: str) -> ChatEngine
```

All return `self` for method chaining. History is automatically trimmed when it exceeds `max_history` (system messages are always preserved).

### ChatEngine.complete()

```python
def complete(
    self,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **kwargs,
) -> str
```

Generates a chat completion. Tries the OpenAI-compatible `/v1/chat/completions` endpoint first, then falls back to prompt-based completion via `engine.infer()`. Automatically adds the assistant's response to history.

**Returns:** Assistant's response text.

```python
chat = ChatEngine(engine, system_prompt="You are helpful.")
chat.add_user_message("Explain photosynthesis")
response = chat.complete()
print(response)
```

### ChatEngine.complete_stream()

```python
def complete_stream(
    self,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **kwargs,
) -> Iterator[str]
```

Streaming chat completion via SSE. Falls back to non-streaming if the streaming endpoint is unavailable.

**Yields:** Text chunks as they are generated.

```python
for chunk in chat.complete_stream():
    print(chunk, end='', flush=True)
```

### ChatEngine.clear_history()

```python
def clear_history(self, keep_system: bool = True) -> ChatEngine
```

### ChatEngine.get_history()

```python
def get_history(self) -> List[Dict[str, str]]
```

**Returns:** List of message dicts in OpenAI format.

### ChatEngine.save_history() / ChatEngine.load_history()

```python
def save_history(self, filepath: str) -> None
def load_history(self, filepath: str) -> ChatEngine
```

Persists conversation to JSON including messages and metadata (max_history, max_tokens, temperature).

### ChatEngine.count_tokens()

```python
def count_tokens(self) -> int
```

Estimates token count for the current conversation. Uses the `/tokenize` endpoint if available, otherwise approximates at 1 token per 4 characters.

---

## ConversationManager

Manages multiple named conversation sessions, allowing switching between contexts.

### ConversationManager(engine)

| Parameter | Type | Description |
|-----------|------|-------------|
| `engine` | `InferenceEngine` | Shared InferenceEngine instance |

### ConversationManager Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `create_conversation(name, system_prompt, **kwargs)` | `str, Optional[str]` | `ChatEngine` | Create a new conversation |
| `switch_to(name)` | `str` | `ChatEngine` | Switch active conversation |
| `get_current()` | -- | `ChatEngine` | Get active conversation |
| `chat(message, **kwargs)` | `str` | `str` | Send message to active conversation |
| `list_conversations()` | -- | `List[str]` | List all conversation names |
| `delete_conversation(name)` | `str` | `None` | Delete a conversation |
| `save_all(directory)` | `str` | `None` | Save all conversations to directory |
| `load_all(directory)` | `str` | `None` | Load all conversations from directory |

```python
manager = ConversationManager(engine)
manager.create_conversation("coding", "You are a coding assistant.")
manager.create_conversation("writing", "You are a writing coach.")
manager.switch_to("coding")
response = manager.chat("How do I write a Python decorator?")
manager.save_all("conversations/")
```

---

## EmbeddingEngine

Text embedding generation with caching and support for both OpenAI-compatible and native llama-server endpoints.

### EmbeddingEngine(engine, pooling, normalize, cache_size)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine` | `InferenceEngine` | -- | InferenceEngine instance |
| `pooling` | `str` | `"mean"` | Pooling strategy (`"mean"`, `"cls"`, `"last"`) |
| `normalize` | `bool` | `True` | Normalize embeddings to unit vectors |
| `cache_size` | `int` | `1000` | Maximum embeddings to cache (FIFO eviction) |

### EmbeddingEngine.embed()

```python
def embed(self, text: str, use_cache: bool = True) -> np.ndarray
```

Generates embedding for a single text. Tries `/v1/embeddings` first, then `/embedding`.

**Returns:** 1D numpy float32 array.

### EmbeddingEngine.embed_batch()

```python
def embed_batch(
    self,
    texts: List[str],
    use_cache: bool = True,
    show_progress: bool = False,
) -> np.ndarray
```

**Returns:** 2D numpy array of shape `(n_texts, embedding_dim)`.

### EmbeddingEngine Cache Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `clear_cache()` | `None` | Clear cache and reset stats |
| `get_cache_stats()` | `Dict` | Returns `cache_size`, `cache_max`, `hits`, `misses`, `hit_rate` |
| `save_cache(filepath)` | `None` | Persist cache to JSON |
| `load_cache(filepath)` | `None` | Load cache from JSON |

```python
embedder = EmbeddingEngine(engine, normalize=True)
vec = embedder.embed("Hello world")
print(f"Dimension: {len(vec)}")
print(embedder.get_cache_stats())
```

---

## Similarity Functions

### cosine_similarity()

```python
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float
```

**Returns:** Cosine similarity (0 to 1 for normalized vectors).

### euclidean_distance()

```python
def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float
```

**Returns:** Euclidean distance between vectors.

### dot_product_similarity()

```python
def dot_product_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float
```

**Returns:** Dot product (equivalent to cosine similarity for unit vectors).

---

## SemanticSearch

Vector similarity search over a document index.

### SemanticSearch(embedder)

| Parameter | Type | Description |
|-----------|------|-------------|
| `embedder` | `EmbeddingEngine` | Embedding engine for vectorization |

### SemanticSearch.add_documents()

```python
def add_documents(
    self,
    documents: List[str],
    metadata: Optional[List[Dict[str, Any]]] = None,
    show_progress: bool = False,
)
```

Embeds documents and adds them to the search index. Can be called multiple times to incrementally build the index.

### SemanticSearch.search()

```python
def search(
    self,
    query: str,
    top_k: int = 5,
    similarity_fn: str = "cosine",
) -> List[Tuple[str, float, Dict[str, Any]]]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | -- | Search query text |
| `top_k` | `int` | `5` | Number of results |
| `similarity_fn` | `str` | `"cosine"` | `"cosine"`, `"dot"`, or `"euclidean"` |

**Returns:** List of `(document_text, score, metadata)` tuples, sorted by descending similarity.

```python
search = SemanticSearch(embedder)
search.add_documents([
    "Python is a programming language",
    "Machine learning uses neural networks",
    "Natural language processing handles text",
])
results = search.search("What is NLP?", top_k=2)
for doc, score, meta in results:
    print(f"{score:.3f}: {doc}")
```

### SemanticSearch.save_index() / load_index() / clear_index()

```python
def save_index(self, filepath: str) -> None
def load_index(self, filepath: str) -> None
def clear_index(self) -> None
```

---

## TextClustering

K-means text clustering using embeddings.

### TextClustering(embedder, n_clusters)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedder` | `EmbeddingEngine` | -- | Embedding engine |
| `n_clusters` | `int` | `5` | Number of clusters |

Requires `scikit-learn`.

### TextClustering.fit()

```python
def fit(self, texts: List[str], show_progress: bool = False) -> np.ndarray
```

**Returns:** Cluster label array of shape `(n_texts,)`.

### TextClustering.get_clusters()

```python
def get_clusters(self, texts: List[str], labels: np.ndarray) -> Dict[int, List[str]]
```

**Returns:** Dict mapping cluster ID to list of texts in that cluster.

### TextClustering.predict()

```python
def predict(self, texts: List[str]) -> np.ndarray
```

Assigns new texts to nearest cluster centers. Raises `RuntimeError` if `fit()` has not been called.

```python
clustering = TextClustering(embedder, n_clusters=3)
labels = clustering.fit(texts)
clusters = clustering.get_clusters(texts, labels)
for cluster_id, docs in clusters.items():
    print(f"Cluster {cluster_id}: {len(docs)} documents")
```

---

## Related Documentation

- [Core API](core-api.md) -- InferenceEngine
- [Kaggle API](kaggle-api.md) -- Kaggle-specific setup
- [Graphistry API](graphistry-api.md) -- Graph visualization
- [Jupyter Workflows Guide](../guides/jupyter-workflows.md)
