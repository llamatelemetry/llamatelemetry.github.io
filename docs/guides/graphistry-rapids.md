# Graphistry and RAPIDS Guide

`llamatelemetry.graphistry` provides adapters and workload helpers for graph-centric analysis.

## Modules

- `workload.py`: split-GPU graph workloads and graph construction
- `connector.py`: Graphistry registration and plotting integration
- `rapids.py`: RAPIDS capability checks and helper wrappers
- `viz.py`: trace and metric visualization classes

## Basic registration

```python
from llamatelemetry.graphistry import register_graphistry

register_graphistry(
    api=3,
    server="hub.graphistry.com",
    personal_key_id="...",
    personal_key_secret="...",
)
```

## Create graph from model output

```python
from llamatelemetry.graphistry import create_graph_from_llm_output

graph = create_graph_from_llm_output(text_output)
```

## RAPIDS checks

```python
from llamatelemetry.graphistry import check_rapids_available

availability = check_rapids_available()
print(availability)
```

## Typical split-GPU pattern

1. Run inference on GPU 0.
2. Move extracted entities/relations to GPU 1 context.
3. Build and plot graph objects with Graphistry/RAPIDS.

See also:

- [Kaggle Environment guide](kaggle-environment.md)
- [Louie Knowledge Graph guide](louie-knowledge-graphs.md)
