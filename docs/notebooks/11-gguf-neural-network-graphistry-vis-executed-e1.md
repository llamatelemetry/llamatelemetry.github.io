# 11 GGUF Neural Network Visualization

Source: `notebooks/11-gguf-neural-network-graphistry-vis-executed-e1.ipynb`


## Notebook focus

This page is a cell-by-cell walkthrough of the notebook, explaining the intent of each step and showing the exact code executed.


## Cell-by-cell walkthrough

### Cell 1 (Markdown)

# 11 GGUF Neural Network Visualization

Use GGUF metadata and `GraphistryBuilders.embedding_knn()` for quick
visual exploration of model structure.

**What you will learn:**
- Extract metadata from a GGUF file with `gguf_report()`
- Build a k-nearest-neighbor graph from embedding vectors
- Visualize token embedding neighborhoods

**Requirements:** llamatelemetry installed. A GGUF model dataset for the report.

### Cell 2 (Markdown)

## 1) Install

### Cell 3 (Code)

**Summary:** Installs required dependencies and runtime tools.


```python
!pip -q install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

### Cell 4 (Markdown)

## 2) GGUF metadata report

### Cell 5 (Code)

**Summary:** Imports core libraries: json, llamatelemetry. Works with GGUF models, quantization, or metadata.


```python
import json
from llamatelemetry.api.gguf import gguf_report

model_path = "/kaggle/input/your-model/model.gguf"

report = gguf_report(model_path)
print(f"Report keys: {list(report.keys())}")
print(json.dumps(report, indent=2, default=str))
```

### Cell 6 (Markdown)

## 3) Build an embedding kNN graph

`embedding_knn()` computes pairwise cosine distances and connects each
node to its k nearest neighbors.

### Cell 7 (Code)

**Summary:** Imports core libraries: llamatelemetry, numpy. Sets up Graphistry for graph visualization or analytics.


```python
import numpy as np
from llamatelemetry.graphistry import GraphistryBuilders

# Simulated embeddings (replace with real token embeddings)
np.random.seed(42)
embeddings = np.random.randn(20, 16).tolist()
labels = [f"token-{i}" for i in range(20)]

nodes_df, edges_df = GraphistryBuilders.embedding_knn(
    embeddings,
    labels=labels,
    k=3,
    metric="cosine",
)

print(f"Nodes: {len(nodes_df)}")
print(nodes_df.head())
print(f"\nEdges: {len(edges_df)}")
print(edges_df.head())
```

### Cell 8 (Markdown)

## 4) Inspect neighborhood structure

### Cell 9 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
# Show neighbors for a specific token
token_id = "token-0"
neighbors = edges_df[edges_df["source"] == token_id]
print(f"Neighbors of {token_id}:")
print(neighbors.to_string(index=False))
```

### Cell 10 (Markdown)

## 5) Visualize (optional)

### Cell 11 (Code)

**Summary:** Sets up Graphistry for graph visualization or analytics.


```python
# from llamatelemetry.graphistry import GraphistrySession
# session = GraphistrySession.from_kaggle_secrets()
# g = session.connector.create_graph(edges_df, nodes_df=nodes_df)
# g.plot()
```
