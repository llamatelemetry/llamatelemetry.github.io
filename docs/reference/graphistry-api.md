# Graphistry API Reference

`llamatelemetry.graphistry` provides GPU-accelerated graph visualization with PyGraphistry and RAPIDS.
It is designed for split-GPU architecture on Kaggle (GPU 0 for LLM inference, GPU 1 for graph operations)
and includes builders for common LLM graph patterns, connectors for the Graphistry Hub service,
RAPIDS-backed graph analytics, and high-level visualization utilities.

```python
from llamatelemetry.graphistry import (
    GraphistryBuilders, InferenceRecord, records_to_dataframe,
    traces_to_records, build_graph_nodes_edges, build_latency_time_series,
    GraphistryViz, TraceVisualization, MetricsVisualization, create_graph_viz,
    RAPIDSBackend, check_rapids_available, create_cudf_dataframe, run_cugraph_algorithm,
    GraphistryConnector, GraphistrySession, register_graphistry, plot_graph,
    GraphWorkload, SplitGPUManager, create_graph_from_llm_output, visualize_knowledge_graph,
)
```

---

## GraphistryBuilders

Static helper methods that build `(nodes_df, edges_df)` pairs suitable for Graphistry visualization.

### GraphistryBuilders.knowledge_graph()

```python
@staticmethod
def knowledge_graph(
    entities: Optional[List[Any]] = None,
    relationships: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entities` | `Optional[List[Any]]` | `None` | Strings or dicts with `id`/`name`/`label` fields |
| `relationships` | `Optional[List[Dict]]` | `None` | Dicts with `source`, `target`, `type` |

**Returns:** Tuple of `(nodes_df, edges_df)` as Pandas DataFrames. If entities are omitted, nodes are derived from relationship endpoints.

```python
nodes_df, edges_df = GraphistryBuilders.knowledge_graph(
    entities=["Python", "AI", "TensorFlow"],
    relationships=[
        {"source": "Python", "target": "AI", "type": "used_for"},
        {"source": "TensorFlow", "target": "AI", "type": "implements"},
    ],
)
```

### GraphistryBuilders.document_similarity()

```python
@staticmethod
def document_similarity(
    documents: List[Any],
    similarities: List[Dict[str, Any]],
    doc_id_key: str = "id",
) -> Tuple[pd.DataFrame, pd.DataFrame]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `documents` | `List[Any]` | -- | Strings or dicts representing documents |
| `similarities` | `List[Dict]` | -- | Dicts with `source`, `target`, and similarity score fields |
| `doc_id_key` | `str` | `"id"` | Key for document ID in dict documents |

**Returns:** Tuple of `(nodes_df, edges_df)`.

### GraphistryBuilders.embedding_knn()

```python
@staticmethod
def embedding_knn(
    embeddings: List[List[float]],
    labels: Optional[List[str]] = None,
    k: int = 5,
    metric: str = "cosine",
) -> Tuple[pd.DataFrame, pd.DataFrame]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embeddings` | `List[List[float]]` | -- | Embedding vectors |
| `labels` | `Optional[List[str]]` | `None` | Labels for nodes (auto-generated if `None`) |
| `k` | `int` | `5` | Number of nearest neighbors |
| `metric` | `str` | `"cosine"` | Distance metric (`"cosine"` or `"euclidean"`) |

**Returns:** Tuple of `(nodes_df, edges_df)` where edges connect each point to its k nearest neighbors with a `score` column.

### GraphistryBuilders.attention_graph()

```python
@staticmethod
def attention_graph(
    attention: List[List[float]],
    tokens: Optional[List[str]] = None,
    threshold: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `attention` | `List[List[float]]` | -- | Attention weight matrix (N x N) |
| `tokens` | `Optional[List[str]]` | `None` | Token strings for labels |
| `threshold` | `float` | `0.0` | Minimum attention weight to include as edge |

**Returns:** Tuple of `(nodes_df, edges_df)` with `weight` column on edges.

---

## InferenceRecord

Normalized inference record dataclass representing one request.

```python
@dataclass
class InferenceRecord:
    ts: float                           # Unix timestamp
    operation: str                      # Operation name (e.g., "chat")
    model: str                          # Model identifier
    latency_ms: float                   # End-to-end latency
    input_tokens: Optional[int]         # Prompt tokens
    output_tokens: Optional[int]        # Generated tokens
    ttfb_ms: Optional[float]            # Time to first byte
    prompt_ms: Optional[float]          # Prompt processing time
    generation_ms: Optional[float]      # Token generation time
    gpu_id: Optional[int]               # GPU device ID
    split_mode: Optional[str]           # Multi-GPU split mode
    success: Optional[bool]             # Request success
    error_type: Optional[str]           # Error type if failed
```

---

## traces_to_records()

```python
def traces_to_records(spans: List[Dict[str, Any]]) -> List[InferenceRecord]
```

Converts exported OpenTelemetry span JSON into `InferenceRecord` objects. Handles both dict-style and OTLP array-style attribute formats. Maps standard `gen_ai.*` and `llm.*` attributes to record fields.

| Parameter | Type | Description |
|-----------|------|-------------|
| `spans` | `List[Dict]` | Span dicts with `start_time_unix_nano`, `end_time_unix_nano`, `attributes` |

**Returns:** List of `InferenceRecord` instances.

---

## records_to_dataframe()

```python
def records_to_dataframe(records: Iterable[InferenceRecord]) -> pd.DataFrame
```

Converts `InferenceRecord` objects into a Pandas DataFrame for analysis and visualization.

---

## build_graph_nodes_edges()

```python
def build_graph_nodes_edges(
    df: pd.DataFrame,
    *,
    node_id_col: str = "operation",
    group_col: str = "model",
) -> Tuple[pd.DataFrame, pd.DataFrame]
```

Builds a directed sequence graph from a DataFrame sorted by timestamp. Nodes represent unique operation-model pairs; edges connect consecutive operations.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | -- | DataFrame with `ts`, operation, and model columns |
| `node_id_col` | `str` | `"operation"` | Column for node identity |
| `group_col` | `str` | `"model"` | Column for grouping |

**Returns:** Tuple of `(nodes_df, edges_df)` with `id`, `label`, `group`, `count` on nodes and `src`, `dst`, `weight` on edges.

---

## build_latency_time_series()

```python
def build_latency_time_series(df: pd.DataFrame, *, bucket: str = "1min") -> pd.DataFrame
```

Aggregates latency into a time series with p50, p95, and count columns.

**Returns:** DataFrame with columns `time`, `latency_ms_p50`, `latency_ms_p95`, `count`.

---

## GraphistryViz

High-level visualization builder for LLM telemetry data.

```python
class GraphistryViz:
    is_registered: bool  # property
```

### GraphistryViz(auto_register=True)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_register` | `bool` | `True` | Try to register using environment variables or Kaggle secrets |

Raises `ImportError` if `pygraphistry` or `pandas` is not installed.

### GraphistryViz.plot_inference_results()

```python
def plot_inference_results(
    self,
    results: List[Any],
    color_by: str = "latency_ms",
    size_by: str = "tokens_generated",
    title: str = "Inference Results",
    **kwargs,
)
```

Creates a sequential graph where each `InferResult` is a node, colored and sized by the specified attributes. Extracts `latency_ms`, `tokens_generated`, `tokens_per_sec`, and `success` from result objects.

### GraphistryViz.plot_trace_graph()

```python
def plot_trace_graph(
    self,
    spans: List[Dict[str, Any]],
    color_by: str = "duration_ms",
    size_by: str = "tokens",
    title: str = "Trace Graph",
    **kwargs,
)
```

Plots OpenTelemetry spans as a directed parent-child graph using `parent_span_id` relationships. Returns `None` if no parent-child relationships exist.

### GraphistryViz.plot_gpu_metrics()

```python
def plot_gpu_metrics(
    self,
    metrics: List[Dict[str, Any]],
    time_column: str = "timestamp",
    color_by: str = "gpu_utilization",
    size_by: str = "memory_used",
    title: str = "GPU Metrics Timeline",
    **kwargs,
)
```

Plots GPU metrics over time as a timeline graph.

### GraphistryViz.plot_latency_distribution()

```python
def plot_latency_distribution(
    self,
    results: List[Any],
    bins: int = 20,
    title: str = "Latency Distribution",
    **kwargs,
)
```

Plots a histogram-style graph of latency values, colored by bin center and sized by count.

### GraphistryViz.plot_knowledge_graph()

```python
def plot_knowledge_graph(
    self,
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    entity_id_col: str = "id",
    entity_label_col: str = "name",
    rel_source_col: str = "source",
    rel_target_col: str = "target",
    rel_type_col: str = "type",
    title: str = "Knowledge Graph",
    **kwargs,
)
```

Plots a knowledge graph from entities and relationships. Labels nodes and color-codes edges by relationship type.

---

## TraceVisualization / MetricsVisualization

Configuration dataclasses for visualization settings.

```python
@dataclass
class TraceVisualization:
    color_by: str = "latency_ms"
    size_by: str = "tokens"
    layout: str = "force"
    title: str = "LLM Inference Traces"
    palette: str = "viridis"

@dataclass
class MetricsVisualization:
    metric: str = "latency_ms"
    aggregation: str = "mean"
    time_window: str = "1min"
```

---

## create_graph_viz()

```python
def create_graph_viz(
    edges: pd.DataFrame,
    nodes: Optional[pd.DataFrame] = None,
    source: str = "source",
    target: str = "target",
    node_id: str = "id",
    color_by: Optional[str] = None,
    size_by: Optional[str] = None,
    title: str = "Graph",
    auto_register: bool = True,
    **kwargs,
)
```

Convenience function for quick graph visualization from DataFrames. Auto-registers with Graphistry if credentials are available.

---

## RAPIDSBackend

Unified interface for GPU-accelerated graph analytics with cuDF, cuGraph, and cuML.

```python
class RAPIDSBackend:
    cudf_available: bool      # property
    cugraph_available: bool   # property
    cuml_available: bool      # property
```

### RAPIDSBackend(gpu_id=1)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpu_id` | `int` | `1` | GPU device to use (sets `CUDA_VISIBLE_DEVICES`) |

Raises `ImportError` if no RAPIDS component is installed.

### RAPIDSBackend Methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `create_dataframe(data)` | `data: Union[Dict, List[Dict]]` | `cudf.DataFrame` |
| `from_pandas(pdf)` | `pdf: pd.DataFrame` | `cudf.DataFrame` |
| `to_pandas(gdf)` | `gdf: cudf.DataFrame` | `pd.DataFrame` |
| `pagerank(edges_df, source_col="src", dest_col="dst", damping=0.85, max_iter=100)` | -- | DataFrame with `vertex`, `pagerank` columns |
| `louvain(edges_df, source_col="src", dest_col="dst", weight_col=None, resolution=1.0)` | -- | Tuple of `(partitions_df, modularity)` |
| `betweenness_centrality(edges_df, source_col="src", dest_col="dst", k=None, normalized=True)` | -- | DataFrame with `vertex`, `betweenness_centrality` |
| `connected_components(edges_df, source_col="src", dest_col="dst")` | -- | DataFrame with `vertex`, `labels` |
| `umap(data, n_components=2, n_neighbors=15, min_dist=0.1)` | -- | Reduced embeddings array |

```python
backend = RAPIDSBackend(gpu_id=1)
gdf = backend.create_dataframe({"src": [0, 1, 2], "dst": [1, 2, 0]})
pr = backend.pagerank(gdf)
```

---

## check_rapids_available()

```python
def check_rapids_available() -> Dict[str, bool]
```

**Returns:** Dict with keys `"cudf"`, `"cugraph"`, `"cuml"`, `"pylibraft"` indicating availability.

---

## run_cugraph_algorithm()

```python
def run_cugraph_algorithm(
    algorithm: str,
    edges_df,
    source_col: str = "src",
    dest_col: str = "dst",
    **kwargs,
) -> Any
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `algorithm` | `str` | Algorithm name: `"pagerank"`, `"louvain"`, `"betweenness_centrality"`, `"connected_components"` |
| `edges_df` | DataFrame | Edge DataFrame |
| `source_col` | `str` | Source column name |
| `dest_col` | `str` | Destination column name |

---

## GraphistryConnector

Connector for the Graphistry visualization service with registration and graph creation.

### GraphistryConnector(auto_register=True, server="hub.graphistry.com")

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_register` | `bool` | `True` | Try to register using environment variables |
| `server` | `str` | `"hub.graphistry.com"` | Graphistry server URL |

### GraphistryConnector Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `register(username, password, api_key)` | Register with Graphistry | `bool` |
| `create_graph(edges_df, source, destination, nodes_df, node_id)` | Create graph object | Graphistry plotter |
| `plot(edges_df, source, destination, nodes_df, node_id, **kwargs)` | Quick plot | Plot result |
| `compute_igraph(edges_df, source, destination, algorithm)` | Run igraph algorithm | Graphistry graph |
| `compute_cugraph(edges_df, source, destination, algorithm)` | Run cuGraph algorithm (GPU) | Graphistry graph |

```python
connector = GraphistryConnector()
g = connector.create_graph(edges_df, source="src", destination="dst", nodes_df=nodes_df)
g.plot()
```

---

## register_graphistry()

```python
def register_graphistry(
    username: Optional[str] = None,
    password: Optional[str] = None,
    api_key: Optional[str] = None,
    server: str = "hub.graphistry.com",
) -> bool
```

Module-level function. Uses environment variables `GRAPHISTRY_USERNAME`, `GRAPHISTRY_PASSWORD`, `GRAPHISTRY_API_KEY` if credentials are not provided directly.

---

## GraphistrySession

Helper dataclass for creating registered Graphistry sessions.

```python
@dataclass
class GraphistrySession:
    connector: GraphistryConnector
    registered: bool
    server: str = "hub.graphistry.com"
```

### GraphistrySession.from_kaggle_secrets()

```python
@classmethod
def from_kaggle_secrets(cls, server="hub.graphistry.com", auto_register=True) -> GraphistrySession
```

### GraphistrySession.from_env()

```python
@classmethod
def from_env(cls, server="hub.graphistry.com", auto_register=True) -> GraphistrySession
```

---

## GraphWorkload

GPU-accelerated graph workload manager for RAPIDS and Graphistry.

### GraphWorkload(gpu_id, graphistry_username, graphistry_password, graphistry_server)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpu_id` | `int` | `1` | GPU for graph operations |
| `graphistry_username` | `Optional[str]` | `None` | Graphistry Hub username |
| `graphistry_password` | `Optional[str]` | `None` | Graphistry Hub password |
| `graphistry_server` | `str` | `"hub.graphistry.com"` | Graphistry server URL |

### GraphWorkload.create_knowledge_graph()

```python
def create_knowledge_graph(
    self,
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    use_gpu: bool = True,
) -> Any
```

Creates a Graphistry graph with styled nodes (colored by entity type) and edges (colored by weight).

### GraphWorkload.run_pagerank() / GraphWorkload.run_community_detection()

GPU-accelerated PageRank and Louvain community detection via cuGraph.

---

## SplitGPUManager

Manages GPU assignments for split LLM/Graph workloads on dual-GPU systems.

```python
manager = SplitGPUManager(auto_detect=True)
print(manager.get_graph_env())  # {"CUDA_VISIBLE_DEVICES": "1"}
print(manager.get_llm_env())    # {"CUDA_VISIBLE_DEVICES": "0"}
args = manager.get_llama_server_args("/path/to/model.gguf")
```

---

## create_graph_from_llm_output()

```python
def create_graph_from_llm_output(
    llm_response: str,
    workload: Optional[GraphWorkload] = None,
) -> Any
```

Parses JSON with `entities` and `relationships` keys from LLM output text and returns a Graphistry graph. Raises `ValueError` if no valid JSON is found.

## visualize_knowledge_graph()

```python
def visualize_knowledge_graph(
    entities: List[Dict], relationships: List[Dict], gpu_id: int = 1, **kwargs
) -> Any
```

One-liner for knowledge graph visualization. Creates a `GraphWorkload` and plots the graph.

---

## Related Documentation

- [Louie API](louie-api.md) -- AI-powered graph analysis
- [Kaggle API](kaggle-api.md) -- GPU context management
- [CUDA & Inference API](cuda-inference-api.md) -- Low-level GPU optimization
- [Graphistry and RAPIDS Guide](../guides/graphistry-rapids.md)
