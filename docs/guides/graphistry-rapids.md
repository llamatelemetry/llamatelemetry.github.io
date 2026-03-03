# Graphistry and RAPIDS

llamatelemetry integrates with Graphistry for interactive graph visualization and RAPIDS cuGraph for GPU-accelerated graph analytics. This module enables visual exploration of inference traces, knowledge graphs, document similarity networks, and embedding spaces.

## Overview

The Graphistry and RAPIDS integration provides:

- **GraphistrySession** -- manages authentication and connection to Graphistry Hub
- **GraphistryBuilders** -- pre-built graph constructors for common patterns
- **Trace-to-Graph** -- converts OpenTelemetry traces into graph structures
- **GraphistryViz** -- renders interactive graph visualizations
- **RAPIDSBackend** -- GPU-accelerated graph algorithms (PageRank, Louvain, UMAP)
- **SplitGPUManager** -- coordinates GPU allocation between LLM and graph workloads
- **GraphWorkload** -- tracks and manages graph computation tasks

## GraphistrySession

### Authentication

```python
from llamatelemetry.graphistry.connector import GraphistrySession

# From explicit credentials
session = GraphistrySession(
    server="https://hub.graphistry.com",
    username="your-username",
    password="your-password",
)
session.login()
```

### From Kaggle Secrets

```python
session = GraphistrySession.from_kaggle_secrets()
# Reads GRAPHISTRY_USERNAME, GRAPHISTRY_PASSWORD, GRAPHISTRY_SERVER
# from Kaggle notebook secrets
```

### Session Methods

| Method | Description |
|--------|-------------|
| `login()` | Authenticate with Graphistry server |
| `is_authenticated()` | Check if session is active |
| `register(edges_df, nodes_df)` | Register a graph for visualization |
| `plot()` | Render the graph in the notebook |

## GraphistryBuilders

Pre-built graph constructors for common LLM observability patterns:

### Knowledge Graph

Visualize entities and relationships extracted from text:

```python
from llamatelemetry.graphistry.builders import GraphistryBuilders

# Assuming you have a KnowledgeGraph from LouieClient
from llamatelemetry.louie.knowledge import KnowledgeExtractor

extractor = KnowledgeExtractor()
kg = extractor.extract("NVIDIA created CUDA for parallel computing on GPUs.")

# Build Graphistry-compatible graph
nodes_df, edges_df = GraphistryBuilders.knowledge_graph(kg)

# Visualize
session = GraphistrySession.from_kaggle_secrets()
session.login()
g = session.register(edges_df, nodes_df)
g.plot()
```

### Document Similarity

Build a similarity graph from document embeddings:

```python
documents = [
    "Flash attention reduces memory usage.",
    "CUDA enables parallel GPU computing.",
    "GGUF stores quantized model weights.",
    "Attention is a key transformer component.",
]
embeddings = [...]  # List of embedding vectors

nodes_df, edges_df = GraphistryBuilders.document_similarity(
    documents=documents,
    embeddings=embeddings,
    threshold=0.7,  # Minimum similarity to create an edge
)
```

### Embedding KNN Graph

Build a K-nearest-neighbors graph from embedding vectors:

```python
labels = ["doc_1", "doc_2", "doc_3", "doc_4"]
embeddings = [...]  # numpy arrays

nodes_df, edges_df = GraphistryBuilders.embedding_knn(
    labels=labels,
    embeddings=embeddings,
    k=3,  # Number of nearest neighbors
)
```

### Attention Graph

Visualize attention patterns from transformer layers:

```python
# attention_matrix is a numpy array of shape (n_tokens, n_tokens)
tokens = ["[CLS]", "What", "is", "CUDA", "?", "[SEP]"]
attention_matrix = [...]

nodes_df, edges_df = GraphistryBuilders.attention_graph(
    tokens=tokens,
    attention_weights=attention_matrix,
    threshold=0.1,  # Minimum attention weight to show
)
```

## Trace-to-Graph Pipeline

Convert OpenTelemetry traces into graph structures for analysis:

### Step 1: Collect Trace Records

```python
from llamatelemetry.graphistry.viz import traces_to_records

# Get trace records from the telemetry exporter
records = traces_to_records(span_data)
# Returns: list of dicts with trace_id, span_id, parent_id, name, duration, attributes
```

### Step 2: Convert to DataFrame

```python
from llamatelemetry.graphistry.viz import records_to_dataframe

df = records_to_dataframe(records)
print(df.columns)
# ['trace_id', 'span_id', 'parent_span_id', 'name', 'duration_ms',
#  'start_time', 'end_time', 'gen_ai.request.model', ...]
```

### Step 3: Build Graph Structure

```python
from llamatelemetry.graphistry.viz import build_graph_nodes_edges

nodes_df, edges_df = build_graph_nodes_edges(df)
# nodes: span_id, name, duration_ms, attributes
# edges: source (parent_span_id), target (span_id), relationship
```

### Step 4: Visualize

```python
session = GraphistrySession.from_kaggle_secrets()
session.login()
g = session.register(edges_df, nodes_df)
g.plot()
```

### Latency Time Series

Build a time-series visualization of inference latencies:

```python
from llamatelemetry.graphistry.viz import build_latency_time_series

ts_df = build_latency_time_series(df)
# DataFrame with timestamp, latency_ms, tokens_per_sec, model columns
# Suitable for plotting with matplotlib or Graphistry
```

## RAPIDSBackend

GPU-accelerated graph algorithms via RAPIDS cuGraph:

```python
from llamatelemetry.graphistry.rapids import RAPIDSBackend

backend = RAPIDSBackend()
```

### PageRank

Find the most important nodes in the graph:

```python
pagerank_scores = backend.pagerank(edges_df, source="src", target="dst")
print(pagerank_scores.head())
# Returns DataFrame with node_id and pagerank columns
```

### Louvain Community Detection

Discover communities in the graph:

```python
communities = backend.louvain(edges_df, source="src", target="dst")
print(communities.head())
# Returns DataFrame with node_id and community columns

# Inspect community sizes
community_sizes = communities["community"].value_counts()
print(f"Found {len(community_sizes)} communities")
```

### Betweenness Centrality

Identify bridge nodes that connect different parts of the graph:

```python
centrality = backend.betweenness_centrality(
    edges_df, source="src", target="dst"
)
top_bridges = centrality.nlargest(10, "centrality")
print(top_bridges)
```

### UMAP Dimensionality Reduction

Project high-dimensional embeddings to 2D for visualization:

```python
import numpy as np

embeddings = np.random.randn(100, 768)  # 100 documents, 768 dims

coords_2d = backend.umap(
    embeddings,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
)
# Returns numpy array of shape (100, 2)
```

## SplitGPUManager

Manage GPU allocation between LLM inference and graph analytics:

```python
from llamatelemetry.kaggle.gpu_context import split_gpu_session

# GPU 0 for LLM inference, GPU 1 for RAPIDS/Graphistry
with split_gpu_session(llm_gpu=0, graph_gpu=1):
    # LLM operations use GPU 0
    engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)
    result = engine.infer("What is CUDA?", max_tokens=128)

    # RAPIDS operations automatically use GPU 1
    backend = RAPIDSBackend()
    scores = backend.pagerank(edges_df)
```

## GraphWorkload

Track graph computation workloads:

```python
from llamatelemetry.graphistry.workload import GraphWorkload

workload = GraphWorkload(name="trace-analysis")
workload.add_task("pagerank", edges_df)
workload.add_task("louvain", edges_df)

results = workload.execute(backend)
print(f"Tasks completed: {workload.completed_count}")
print(f"Total time: {workload.total_duration_ms:.1f} ms")
```

## Complete Example

```python
import llamatelemetry as lt
from llamatelemetry.graphistry.connector import GraphistrySession
from llamatelemetry.graphistry.builders import GraphistryBuilders
from llamatelemetry.graphistry.rapids import RAPIDSBackend
from llamatelemetry.louie.knowledge import KnowledgeExtractor
from llamatelemetry.kaggle.gpu_context import split_gpu_session

# Split GPUs
with split_gpu_session(llm_gpu=0, graph_gpu=1):
    # 1. Run inference to generate text
    with lt.InferenceEngine() as engine:
        engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)
        result = engine.infer(
            "Describe the relationship between CUDA, GPUs, and deep learning.",
            max_tokens=256,
        )

    # 2. Extract knowledge graph from generated text
    extractor = KnowledgeExtractor()
    kg = extractor.extract(result.text)

    # 3. Build Graphistry visualization
    nodes_df, edges_df = GraphistryBuilders.knowledge_graph(kg)

    # 4. Run graph analytics
    backend = RAPIDSBackend()
    pagerank = backend.pagerank(edges_df)
    communities = backend.louvain(edges_df)

    print(f"Entities: {len(kg.entities)}")
    print(f"Relationships: {len(kg.relationships)}")
    print(f"Communities found: {communities['community'].nunique()}")

    # 5. Visualize in Graphistry
    session = GraphistrySession.from_kaggle_secrets()
    session.login()
    g = session.register(edges_df, nodes_df)
    g.plot()
```

## Dependencies

| Package | Required For | Install |
|---------|-------------|---------|
| `pygraphistry` | Visualization | `pip install pygraphistry` |
| `pandas` | DataFrames | `pip install pandas` |
| `cudf` | GPU DataFrames | RAPIDS install |
| `cugraph` | Graph algorithms | RAPIDS install |
| `cuml` | UMAP | RAPIDS install |

!!! note "RAPIDS Availability"
    RAPIDS (cudf, cugraph, cuml) requires NVIDIA GPUs and specific CUDA versions. On Kaggle T4 instances, RAPIDS is pre-installed. For local development, see the [RAPIDS installation guide](https://rapids.ai/start.html).

## Best Practices

- **Split GPUs on Kaggle** -- dedicate one GPU to inference and another to RAPIDS.
- **Use thresholds** in similarity graphs to avoid excessive edges.
- **Start with PageRank** for initial graph exploration before running more expensive algorithms.
- **Cache embeddings** -- recompute only when the underlying data changes.
- **Use Graphistry Hub** for sharing interactive visualizations with collaborators.

## Related

- [Louie Knowledge Graphs](louie-knowledge-graphs.md) -- knowledge extraction
- [Telemetry and Observability](telemetry-observability.md) -- trace data source
- [Kaggle Environment](kaggle-environment.md) -- GPU splitting
- [Graphistry API Reference](../reference/graphistry-api.md)
