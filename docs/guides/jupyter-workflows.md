# Jupyter Workflows

The `llamatelemetry.jupyter`, `llamatelemetry.chat`, and `llamatelemetry.embeddings` modules provide notebook-optimized features for interactive LLM development. This guide covers real-time streaming, interactive chat widgets, multi-turn conversation management, text embeddings, semantic search, and visualization helpers.

## Prerequisites

Install the optional dependencies for full Jupyter support:

```bash
pip install ipywidgets tqdm pandas
```

Verify the environment:

```python
from llamatelemetry.jupyter import is_jupyter_available, check_dependencies

print(f"Jupyter available: {is_jupyter_available()}")
print(f"Widgets available: {check_dependencies(require_widgets=True)}")
```

---

## Real-Time Streaming in Notebooks

The `stream_generate` function streams tokens directly into a Jupyter cell as they are generated, giving you immediate visual feedback without waiting for the full response.

```python
from llamatelemetry import InferenceEngine
from llamatelemetry.jupyter import stream_generate

engine = InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)

# Tokens appear in real-time as markdown
text = stream_generate(
    engine,
    "Explain the attention mechanism in transformers",
    max_tokens=256,
    temperature=0.7,
    show_timing=True,   # Display performance stats after generation
    markdown=True,       # Render output as formatted markdown
)
```

When `show_timing=True`, a performance summary appears after generation completes, showing token count, elapsed time, and tokens per second.

!!! tip "Markdown vs Plain Text"
    Set `markdown=False` if you want raw text output in a `<pre>` block. This is useful for code generation tasks where markdown rendering might interfere with formatting.

If you are not running inside Jupyter, `stream_generate` automatically falls back to a non-streaming `engine.infer()` call and prints the result to stdout.

---

## Batch Generation with Progress Bars

For processing multiple prompts, `progress_generate` shows a tqdm progress bar:

```python
from llamatelemetry.jupyter import progress_generate

prompts = [
    "Summarize quantum computing in one sentence.",
    "What is gradient descent?",
    "Explain attention mechanisms briefly.",
    "Define a neural network.",
]

results = progress_generate(
    engine,
    prompts,
    max_tokens=100,
)

for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result}\n")
```

---

## Displaying Performance Metrics

Render engine performance metrics as a formatted table in Jupyter:

```python
from llamatelemetry.jupyter import display_metrics

# Renders as a pandas DataFrame if pandas is installed,
# otherwise falls back to an HTML table
display_metrics(engine, as_dataframe=True)
```

The function flattens nested metric categories into a clean table with Category, Metric, and Value columns.

---

## Temperature Comparison Tool

Explore how sampling temperature affects model output with a side-by-side comparison:

```python
from llamatelemetry.jupyter import compare_temperatures

results = compare_temperatures(
    engine,
    prompt="Write a creative opening line for a novel about AI.",
    temperatures=[0.3, 0.7, 1.0, 1.5],
    max_tokens=100,
)

# results is a dict: {0.3: "text...", 0.7: "text...", ...}
```

Each temperature setting is displayed in a styled card with the generation speed, making it easy to visually compare deterministic vs creative outputs.

---

## Token Visualization

Inspect how text is tokenized by the model:

```python
from llamatelemetry.jupyter import visualize_tokens

# Displays tokens with visual boundaries
visualize_tokens("The quick brown fox jumps over the lazy dog", engine=engine)
```

When an engine is provided, the function calls the `/tokenize` endpoint and renders each token in a bordered span, followed by a total token count.

---

## Interactive Chat Widget

The `ChatWidget` class provides a full chat interface with text input, send/clear buttons, and adjustable model parameters -- all rendered as ipywidgets inside a notebook cell.

```python
from llamatelemetry import InferenceEngine
from llamatelemetry.jupyter import ChatWidget

engine = InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)

chat = ChatWidget(
    engine,
    system_prompt="You are a helpful coding assistant.",
    max_tokens=256,
    temperature=0.7,
)
chat.display()
```

The widget includes:

- **Text area** for typing messages
- **Send button** to submit and generate a response
- **Clear button** to reset conversation history
- **Max Tokens slider** (50-1024) for controlling response length
- **Temperature slider** (0.0-2.0) for controlling randomness

Each message is styled with colored backgrounds -- blue for user messages, gray for assistant responses, and red for errors. Performance metrics (tokens generated, tokens/sec) appear below each response.

!!! warning "ipywidgets Required"
    The `ChatWidget` raises `ImportError` if `ipywidgets` is not installed. Install it with `pip install ipywidgets` and restart the kernel.

---

## Multi-Turn Chat with ChatEngine

For programmatic chat workflows (without widgets), the `ChatEngine` class manages conversation history, token counting, and both streaming and non-streaming completions via the OpenAI-compatible `/v1/chat/completions` endpoint.

### Basic Usage

```python
from llamatelemetry.chat import ChatEngine

chat = ChatEngine(
    engine,
    system_prompt="You are a Python expert.",
    max_history=20,
    max_tokens=256,
    temperature=0.7,
)

# Add messages and get completions
chat.add_user_message("How do I read a CSV file in Python?")
response = chat.complete()
print(response)

# Continue the conversation (history is maintained)
chat.add_user_message("How do I filter rows?")
response = chat.complete()
print(response)
```

### Streaming Chat

```python
for chunk in chat.complete_stream(max_tokens=512):
    print(chunk, end='', flush=True)
```

### Conversation History Management

```python
# Get history as list of dicts
history = chat.get_history()

# Estimate token count
tokens = chat.count_tokens()
print(f"Conversation uses ~{tokens} tokens")

# Save and load conversations
chat.save_history("conversation.json")
chat.load_history("conversation.json")

# Clear history (keep system prompt by default)
chat.clear_history(keep_system=True)
```

### Method Chaining

`ChatEngine` supports fluent method chaining:

```python
response = (
    ChatEngine(engine, system_prompt="You are helpful.")
    .add_user_message("What is Python?")
    .complete()
)
```

---

## Managing Multiple Conversations

The `ConversationManager` lets you maintain several independent conversation contexts and switch between them:

```python
from llamatelemetry.chat import ConversationManager

manager = ConversationManager(engine)

# Create topic-specific conversations
manager.create_conversation("coding", system_prompt="You are a coding assistant.")
manager.create_conversation("writing", system_prompt="You are a writing coach.")

# Chat in a specific context
manager.switch_to("coding")
response = manager.chat("How do I use list comprehensions?")

# Switch contexts
manager.switch_to("writing")
response = manager.chat("How do I write a strong opening paragraph?")

# List and manage
print(manager.list_conversations())  # ['coding', 'writing']

# Persist all conversations
manager.save_all("conversations/")
manager.load_all("conversations/")
```

---

## Text Embeddings

The `EmbeddingEngine` generates vector embeddings from text using the llama-server embedding endpoints, with built-in caching for efficiency.

### Generating Embeddings

```python
from llamatelemetry.embeddings import EmbeddingEngine

embedder = EmbeddingEngine(
    engine,
    pooling="mean",     # Options: mean, cls, last
    normalize=True,     # Normalize to unit vectors
    cache_size=1000,    # Cache up to 1000 embeddings
)

# Single embedding
vector = embedder.embed("Artificial intelligence is transforming industries")
print(f"Dimension: {len(vector)}")  # Model-specific (e.g., 768)

# Batch embedding with progress bar
texts = ["First document", "Second document", "Third document"]
vectors = embedder.embed_batch(texts, show_progress=True)
print(f"Shape: {vectors.shape}")  # (3, 768)
```

### Cache Management

```python
# Check cache performance
stats = embedder.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Cache size: {stats['cache_size']}/{stats['cache_max']}")

# Persist cache to disk
embedder.save_cache("embeddings_cache.json")
embedder.load_cache("embeddings_cache.json")

# Clear cache
embedder.clear_cache()
```

### Similarity Functions

```python
from llamatelemetry.embeddings import cosine_similarity, euclidean_distance, dot_product_similarity

vec1 = embedder.embed("machine learning")
vec2 = embedder.embed("artificial intelligence")
vec3 = embedder.embed("cooking recipes")

print(f"ML vs AI: {cosine_similarity(vec1, vec2):.3f}")       # High
print(f"ML vs Cooking: {cosine_similarity(vec1, vec3):.3f}")   # Low
```

---

## Semantic Search

Build a simple search index over documents using embeddings:

```python
from llamatelemetry.embeddings import SemanticSearch

search = SemanticSearch(embedder)

# Index documents (with optional metadata)
search.add_documents(
    documents=[
        "Python is a versatile programming language",
        "CUDA enables GPU-accelerated computing",
        "Transformers revolutionized NLP",
        "Gradient descent optimizes neural networks",
    ],
    metadata=[
        {"category": "programming"},
        {"category": "gpu"},
        {"category": "nlp"},
        {"category": "ml"},
    ],
    show_progress=True,
)

# Search
results = search.search("What is GPU computing?", top_k=2)
for doc, score, meta in results:
    print(f"  {score:.3f}: [{meta.get('category')}] {doc}")

# Persist index
search.save_index("search_index.json")
search.load_index("search_index.json")

print(f"Indexed documents: {len(search)}")
```

---

## Text Clustering

Group documents into clusters using K-means on embeddings:

```python
from llamatelemetry.embeddings import TextClustering

clustering = TextClustering(embedder, n_clusters=3)

texts = [
    "Python programming", "Java development",
    "Neural networks", "Deep learning",
    "Data visualization", "Chart creation",
]

labels = clustering.fit(texts, show_progress=True)
clusters = clustering.get_clusters(texts, labels)

for cluster_id, docs in clusters.items():
    print(f"\nCluster {cluster_id}:")
    for doc in docs:
        print(f"  - {doc}")

# Predict cluster for new text
new_labels = clustering.predict(["Machine learning models"])
```

!!! tip "scikit-learn Required"
    `TextClustering` requires scikit-learn for K-means. Install with `pip install scikit-learn`.

---

## Best Practices

1. **Use streaming for long outputs** -- `stream_generate` provides a much better user experience than waiting for full generation to complete.
2. **Cache embeddings** -- The `EmbeddingEngine` cache avoids redundant API calls. Save the cache to disk between sessions for persistent performance.
3. **Manage conversation length** -- Set `max_history` on `ChatEngine` to prevent context window overflow. Use `count_tokens()` to monitor usage.
4. **Save conversations** -- Use `save_history()` / `load_history()` for reproducibility and debugging.
5. **Batch process when possible** -- `embed_batch` and `progress_generate` are more efficient than looping over individual calls.

## Related Reference

- [Jupyter, Chat, and Embeddings API Reference](../reference/jupyter-chat-embeddings.md)
- [Inference Engine Guide](inference-engine.md)
- [API Client Guide](api-client.md)
