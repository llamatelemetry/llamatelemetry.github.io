# 13 Token Embedding Visualizer

Source: `notebooks/13-gguf-token-embedding-visualizer-executed-e1.ipynb`


## Notebook focus

This page is a cell-by-cell walkthrough of the notebook, explaining the intent of each step and showing the exact code executed.


## Cell-by-cell walkthrough

### Cell 1 (Markdown)

# 13 Token Embedding Visualizer

Visualize token embedding neighborhoods using kNN graphs.

**What you will learn:**
- Generate token embeddings (simulated or from a model)
- Build kNN graphs with configurable k and distance metric
- Explore embedding clusters

**Requirements:** llamatelemetry installed. No GPU required for graph building.

### Cell 2 (Markdown)

## 1) Install

### Cell 3 (Code)

**Summary:** Installs required dependencies and runtime tools.


```python
!pip -q install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

### Cell 4 (Markdown)

## 2) Generate or load embeddings

In a real workflow, use `LlamaCppClient.embed()` to get real token
embeddings. Here we simulate 30 tokens in 8 dimensions.

### Cell 5 (Code)

**Summary:** Imports core libraries: numpy.


```python
import numpy as np

np.random.seed(42)
n_tokens = 30
embed_dim = 8

# Create 3 clusters of tokens
cluster_centers = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0]], dtype=float)

embeddings = []
labels = []
for i in range(n_tokens):
    cluster = i % 3
    vec = cluster_centers[cluster] + np.random.randn(embed_dim) * 0.3
    embeddings.append(vec.tolist())
    labels.append(f"tok-{i}-c{cluster}")

print(f"Tokens: {n_tokens}, Dimensions: {embed_dim}")
print(f"Sample labels: {labels[:6]}")
```

### Cell 6 (Markdown)

## 3) Build the kNN graph

### Cell 7 (Code)

**Summary:** Imports core libraries: llamatelemetry. Sets up Graphistry for graph visualization or analytics.


```python
from llamatelemetry.graphistry import GraphistryBuilders

nodes_df, edges_df = GraphistryBuilders.embedding_knn(
    embeddings,
    labels=labels,
    k=4,
    metric="cosine",
)

print(f"Nodes: {len(nodes_df)}, Edges: {len(edges_df)}")
print("\nSample edges:")
print(edges_df.head(8).to_string(index=False))
```

### Cell 8 (Markdown)

## 4) Verify cluster structure

Tokens in the same cluster should have more intra-cluster edges than
inter-cluster edges.

### Cell 9 (Code)

**Summary:** Defines helper functions: get_cluster.


```python
def get_cluster(label):
    return label.split("-c")[-1]

intra = 0
inter = 0
for _, row in edges_df.iterrows():
    if get_cluster(row["source"]) == get_cluster(row["target"]):
        intra += 1
    else:
        inter += 1

print(f"Intra-cluster edges: {intra}")
print(f"Inter-cluster edges: {inter}")
print(f"Ratio: {intra / max(inter, 1):.2f}x")
```

### Cell 10 (Markdown)

## 5) Real embeddings from llama.cpp (optional)

Uncomment to use real embeddings from a running llama-server.

### Cell 11 (Code)

**Summary:** Initializes the OpenAI-compatible llama.cpp HTTP client.


```python
# from llamatelemetry.api import LlamaCppClient
#
# client = LlamaCppClient(base_url="http://127.0.0.1:8090")
# texts = ["CUDA", "GPU", "tensor", "matrix", "network", "layer"]
# real_embeddings = client.embed(texts)
#
# nodes_df, edges_df = GraphistryBuilders.embedding_knn(
#     real_embeddings, labels=texts, k=3
# )
# print(edges_df)
```
