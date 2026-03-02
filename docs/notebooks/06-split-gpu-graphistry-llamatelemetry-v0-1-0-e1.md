# 06 Split-GPU + Graphistry

Source: `notebooks/06-split-gpu-graphistry-llamatelemetry-v0-1-0-e1.ipynb`


## Notebook focus

This page is a cell-by-cell walkthrough of the notebook, explaining the intent of each step and showing the exact code executed.


## Cell-by-cell walkthrough

### Cell 1 (Markdown)

# 06 Split-GPU + Graphistry

Run llama.cpp on GPU 0 and Graphistry/RAPIDS graph analytics on GPU 1.

**What you will learn:**
- Use `split_gpu_session` to partition GPUs
- Build knowledge graphs with `GraphistryBuilders`
- Visualize graphs with `GraphistrySession` (optional credentials)

**Requirements:** Kaggle T4 x2. Graphistry credentials (optional) for
visualization.

### Cell 2 (Markdown)

## 1) Install

### Cell 3 (Code)

**Summary:** Installs required dependencies and runtime tools.


```python
!pip -q install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

### Cell 4 (Markdown)

## 2) Split GPU session

The context manager pins the LLM workload to one GPU and reserves the
other for graph/analytics tasks.

### Cell 5 (Code)

**Summary:** Imports core libraries: llamatelemetry.


```python
from llamatelemetry.kaggle import split_gpu_session

with split_gpu_session(llm_gpu=0, graph_gpu=1) as ctx:
    print("LLM server kwargs:", ctx["llm_server_kwargs"])
    print("Graph GPU ID:", ctx.get("graph_gpu"))
```

### Cell 6 (Markdown)

## 3) Build a knowledge graph

`GraphistryBuilders.knowledge_graph()` creates node/edge DataFrames from
entity and relationship lists.

### Cell 7 (Code)

**Summary:** Imports core libraries: llamatelemetry. Sets up Graphistry for graph visualization or analytics.


```python
from llamatelemetry.graphistry import GraphistryBuilders

entities = [
    {"id": "llamatelemetry", "label": "llamatelemetry SDK"},
    {"id": "llama.cpp", "label": "llama.cpp"},
    {"id": "otel", "label": "OpenTelemetry"},
    {"id": "graphistry", "label": "Graphistry"},
]
relationships = [
    {"source": "llamatelemetry", "target": "llama.cpp", "type": "wraps"},
    {"source": "llamatelemetry", "target": "otel", "type": "exports_to"},
    {"source": "llamatelemetry", "target": "graphistry", "type": "visualizes_with"},
]

nodes_df, edges_df = GraphistryBuilders.knowledge_graph(entities, relationships)
print("Nodes:")
print(nodes_df)
print("\nEdges:")
print(edges_df)
```

### Cell 8 (Markdown)

## 4) Visualize with Graphistry (optional)

Requires Graphistry credentials stored in Kaggle Secrets:
- `GRAPHISTRY_USERNAME`
- `GRAPHISTRY_PASSWORD`
- `GRAPHISTRY_SERVER` (optional, defaults to hub.graphistry.com)

### Cell 9 (Code)

**Summary:** Sets up Graphistry for graph visualization or analytics.


```python
# from llamatelemetry.graphistry import GraphistrySession
#
# session = GraphistrySession.from_kaggle_secrets()
# g = session.connector.create_graph(edges_df, nodes_df=nodes_df)
# g.plot()
```
