# 07 Knowledge Graph Extraction

Source: `notebooks/07-knowledge-graph-extraction-graphistry-v0-1-1-e1.ipynb`


## Notebook focus

This page is a cell-by-cell walkthrough of the notebook, explaining the intent of each step and showing the exact code executed.


## Cell-by-cell walkthrough

### Cell 1 (Markdown)

# 07 Knowledge Graph Extraction

Build knowledge graphs from structured entity/relationship data using
`GraphistryBuilders`.

**What you will learn:**
- Model entities with id/label dicts
- Define typed relationships (source, target, type)
- Generate node and edge DataFrames for visualization

**Requirements:** llamatelemetry installed. No GPU required for graph building.

### Cell 2 (Markdown)

## 1) Install

### Cell 3 (Code)

**Summary:** Installs required dependencies and runtime tools.


```python
!pip -q install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1
```

### Cell 4 (Markdown)

## 2) Define a GPU infrastructure knowledge graph

### Cell 5 (Code)

**Summary:** Imports core libraries: llamatelemetry. Sets up Graphistry for graph visualization or analytics.


```python
from llamatelemetry.graphistry import GraphistryBuilders

entities = [
    {"id": "GPU0", "label": "GPU 0 (T4)"},
    {"id": "GPU1", "label": "GPU 1 (T4)"},
    {"id": "llama.cpp", "label": "llama.cpp server"},
    {"id": "nccl", "label": "NCCL"},
    {"id": "otel", "label": "OpenTelemetry"},
]

relationships = [
    {"source": "llama.cpp", "target": "GPU0", "type": "runs_on"},
    {"source": "llama.cpp", "target": "GPU1", "type": "splits_to"},
    {"source": "nccl", "target": "GPU0", "type": "communicates"},
    {"source": "nccl", "target": "GPU1", "type": "communicates"},
    {"source": "otel", "target": "llama.cpp", "type": "observes"},
]

nodes_df, edges_df = GraphistryBuilders.knowledge_graph(entities, relationships)
```

### Cell 6 (Markdown)

## 3) Inspect the DataFrames

### Cell 7 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
print("=== Nodes ===")
print(nodes_df.to_string(index=False))
print(f"\n=== Edges ({len(edges_df)} relationships) ===")
print(edges_df.to_string(index=False))
```

### Cell 8 (Markdown)

## 4) Visualize (optional)

Pass the DataFrames to Graphistry for interactive GPU-accelerated
visualization.

### Cell 9 (Code)

**Summary:** Sets up Graphistry for graph visualization or analytics.


```python
# from llamatelemetry.graphistry import GraphistrySession
# session = GraphistrySession.from_kaggle_secrets()
# g = session.connector.create_graph(edges_df, nodes_df=nodes_df)
# g.plot()
```
