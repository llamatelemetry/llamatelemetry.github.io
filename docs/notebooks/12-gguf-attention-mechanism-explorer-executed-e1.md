# 12 Attention Mechanism Explorer

Source: `notebooks/12-gguf-attention-mechanism-explorer-executed-e1.ipynb`


## Notebook focus

This page is a cell-by-cell walkthrough of the notebook, explaining the intent of each step and showing the exact code executed.


## Cell-by-cell walkthrough

### Cell 1 (Markdown)

# 12 Attention Mechanism Explorer

Visualize attention patterns as directed graphs using
`GraphistryBuilders.attention_graph()`.

**What you will learn:**
- Build attention graphs from attention weight matrices
- Filter edges by attention threshold
- Identify strongly connected token pairs

**Requirements:** llamatelemetry installed. No GPU required.

### Cell 2 (Markdown)

## 1) Install

### Cell 3 (Code)

**Summary:** Installs required dependencies and runtime tools.


```python
!pip -q install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

### Cell 4 (Markdown)

## 2) Simulate an attention matrix

In practice, extract real attention weights from your model. Here we
generate a synthetic 8x8 attention matrix.

### Cell 5 (Code)

**Summary:** Imports core libraries: numpy.


```python
import numpy as np

np.random.seed(42)
seq_len = 8
tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "dog"]

# Simulated softmax attention weights (rows sum to ~1)
raw = np.random.rand(seq_len, seq_len)
attention = (raw / raw.sum(axis=1, keepdims=True)).tolist()

print(f"Attention matrix shape: {seq_len}x{seq_len}")
print(f"Tokens: {tokens}")
```

### Cell 6 (Markdown)

## 3) Build the attention graph

`threshold` filters out weak attention edges. Higher threshold = fewer,
stronger connections.

### Cell 7 (Code)

**Summary:** Imports core libraries: llamatelemetry. Sets up Graphistry for graph visualization or analytics.


```python
from llamatelemetry.graphistry import GraphistryBuilders

nodes_df, edges_df = GraphistryBuilders.attention_graph(
    attention,
    tokens=tokens,
    threshold=0.15,  # keep edges with weight >= 0.15
)

print(f"Nodes: {len(nodes_df)}")
print(nodes_df)
print(f"\nEdges (threshold=0.15): {len(edges_df)}")
print(edges_df.head(10))
```

### Cell 8 (Markdown)

## 4) Find strongest attention pairs

### Cell 9 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
top_edges = edges_df.nlargest(5, "weight")
print("Top 5 attention edges:")
print(top_edges.to_string(index=False))
```

### Cell 10 (Markdown)

## 5) Compare thresholds

### Cell 11 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
for t in [0.0, 0.10, 0.15, 0.20, 0.25]:
    _, e = GraphistryBuilders.attention_graph(attention, tokens=tokens, threshold=t)
    print(f"  threshold={t:.2f} -> {len(e)} edges")
```

### Cell 12 (Markdown)

## 6) Visualize (optional)

### Cell 13 (Code)

**Summary:** Sets up Graphistry for graph visualization or analytics.


```python
# from llamatelemetry.graphistry import GraphistrySession
# session = GraphistrySession.from_kaggle_secrets()
# g = session.connector.create_graph(edges_df, nodes_df=nodes_df)
# g.plot()
```
