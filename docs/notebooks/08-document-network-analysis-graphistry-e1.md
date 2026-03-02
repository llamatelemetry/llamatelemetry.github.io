# 08 Document Network Analysis

Source: `notebooks/08-document-network-analysis-graphistry-e1.ipynb`


## Notebook focus

This page is a cell-by-cell walkthrough of the notebook, explaining the intent of each step and showing the exact code executed.


## Cell-by-cell walkthrough

### Cell 1 (Markdown)

# 08 Document Network Analysis

Build document similarity networks with `GraphistryBuilders.document_similarity()`.

**What you will learn:**
- Model documents as nodes with metadata
- Define pairwise similarity scores as edges
- Generate DataFrames for graph visualization

**Requirements:** llamatelemetry installed. No GPU required.

### Cell 2 (Markdown)

## 1) Install

### Cell 3 (Code)

**Summary:** Installs required dependencies and runtime tools.


```python
!pip -q install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

### Cell 4 (Markdown)

## 2) Define documents and similarity scores

### Cell 5 (Code)

**Summary:** Imports core libraries: llamatelemetry. Sets up Graphistry for graph visualization or analytics.


```python
from llamatelemetry.graphistry import GraphistryBuilders

documents = [
    {"id": "doc1", "title": "CUDA Programming Guide"},
    {"id": "doc2", "title": "llama.cpp Server Architecture"},
    {"id": "doc3", "title": "OpenTelemetry Metrics Specification"},
    {"id": "doc4", "title": "NCCL Collective Operations"},
    {"id": "doc5", "title": "GPU Memory Management"},
]

similarities = [
    {"source": "doc1", "target": "doc2", "score": 0.72},
    {"source": "doc1", "target": "doc4", "score": 0.85},
    {"source": "doc1", "target": "doc5", "score": 0.91},
    {"source": "doc2", "target": "doc3", "score": 0.68},
    {"source": "doc4", "target": "doc5", "score": 0.77},
]
```

### Cell 6 (Markdown)

## 3) Build the similarity graph

### Cell 7 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
nodes_df, edges_df = GraphistryBuilders.document_similarity(documents, similarities)

print("=== Document Nodes ===")
print(nodes_df.to_string(index=False))
print(f"\n=== Similarity Edges ({len(edges_df)}) ===")
print(edges_df.to_string(index=False))
```

### Cell 8 (Markdown)

## 4) Filter by threshold

### Cell 9 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
threshold = 0.75
strong_edges = edges_df[edges_df["score"] >= threshold]
print(f"Edges with score >= {threshold}:")
print(strong_edges.to_string(index=False))
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
