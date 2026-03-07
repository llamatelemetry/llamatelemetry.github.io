# 18 OTel + Graphistry Trace Glue

Source: `notebooks/18-otel-graphistry-trace-glue-e2.ipynb`


## Notebook focus

This page is a cell-by-cell walkthrough of the notebook, explaining the intent of each step and showing the exact code executed.


## Cell-by-cell walkthrough

### Cell 1 (Markdown)

# 18 OTel + Graphistry Trace Glue

Convert OpenTelemetry trace spans into DataFrames and build graph
visualizations with the trace-to-graph helpers.

**What you will learn:**
- Convert span JSON to `InferenceRecord` objects
- Build DataFrames from records
- Generate node/edge DataFrames for graph visualization
- Create latency time series

**Requirements:** llamatelemetry installed. No GPU required.

### Cell 2 (Markdown)

## 1) Install

### Cell 3 (Code)

**Summary:** Installs required dependencies and runtime tools.


```python
!pip -q install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1
```

### Cell 4 (Markdown)

## 2) Import the trace-to-graph helpers

These functions live in `llamatelemetry.graphistry.builders`.

### Cell 5 (Code)

**Summary:** Imports core libraries: llamatelemetry. Sets up Graphistry for graph visualization or analytics.


```python
from llamatelemetry.graphistry.builders import (
    traces_to_records,
    records_to_dataframe,
    build_graph_nodes_edges,
    build_latency_time_series,
)
```

### Cell 6 (Markdown)

## 3) Simulate exported spans

In production, these come from your OTLP exporter's span JSON output.
Here we create synthetic span data.

### Cell 7 (Code)

**Summary:** Imports core libraries: time.


```python
import time

now = time.time()

spans = [
    {
        "name": "chat_completions",
        "start_time": now - 300,
        "end_time": now - 299.5,
        "attributes": {
            "gen_ai.system": "llama.cpp",
            "gen_ai.request.model": "gemma-3-1b-Q4_K_M",
            "gen_ai.usage.input_tokens": 12,
            "gen_ai.usage.output_tokens": 48,
            "gen_ai.response.finish_reasons": ["stop"],
        },
    },
    {
        "name": "chat_completions",
        "start_time": now - 200,
        "end_time": now - 199.3,
        "attributes": {
            "gen_ai.system": "llama.cpp",
            "gen_ai.request.model": "gemma-3-1b-Q4_K_M",
            "gen_ai.usage.input_tokens": 20,
            "gen_ai.usage.output_tokens": 64,
            "gen_ai.response.finish_reasons": ["stop"],
        },
    },
    {
        "name": "embeddings",
        "start_time": now - 100,
        "end_time": now - 99.9,
        "attributes": {
            "gen_ai.system": "llama.cpp",
            "gen_ai.request.model": "gemma-3-1b-Q4_K_M",
            "gen_ai.usage.input_tokens": 5,
            "gen_ai.usage.output_tokens": 0,
        },
    },
]
print(f"Simulated {len(spans)} spans.")
```

### Cell 8 (Markdown)

## 4) Convert spans to InferenceRecords

### Cell 9 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
records = traces_to_records(spans)
print(f"Records: {len(records)}")
for r in records:
    print(f"  {r.operation} | model={r.model} | latency={r.latency_ms:.1f}ms")
```

### Cell 10 (Markdown)

## 5) Build a DataFrame

### Cell 11 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
df = records_to_dataframe(records)
print(f"Columns: {list(df.columns)}")
print(df.to_string(index=False))
```

### Cell 12 (Markdown)

## 6) Build graph nodes and edges

### Cell 13 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
nodes_df, edges_df = build_graph_nodes_edges(df)

print("=== Graph Nodes ===")
print(nodes_df.to_string(index=False))
print("\n=== Graph Edges ===")
print(edges_df.to_string(index=False))
```

### Cell 14 (Markdown)

## 7) Latency time series

### Cell 15 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
ts_df = build_latency_time_series(df, bucket="1min")
print(ts_df.to_string(index=False))
```

### Cell 16 (Markdown)

## 8) Visualize with Graphistry (optional)

### Cell 17 (Code)

**Summary:** Sets up Graphistry for graph visualization or analytics.


```python
# from llamatelemetry.graphistry import GraphistrySession
# session = GraphistrySession.from_kaggle_secrets()
# g = session.connector.create_graph(edges_df, nodes_df=nodes_df)
# g.plot()
```
