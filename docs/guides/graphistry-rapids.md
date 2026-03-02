# Graphistry and RAPIDS

`llamatelemetry.graphistry` provides optional integration for visualizing inference traces, knowledge graphs, and model structures with Graphistry and RAPIDS.

## Modules

- `connector`: Auth and connection helpers
- `viz`: Visualization helpers
- `builders`: Graph construction utilities
- `workload`: Workload summary models
- `rapids`: GPU DataFrame integration

## Typical workflow

1. Build or extract a graph from inference or knowledge extraction
2. Convert to a DataFrame or edge list
3. Use Graphistry to render or share

## Minimal example

```python
from llamatelemetry.graphistry.connector import GraphistryConnector
from llamatelemetry.graphistry.viz import plot_graph

connector = GraphistryConnector()
connector.login()

# Example: plot a simple edge list
edges = [("llama", "telemetry"), ("telemetry", "graphistry")]
plot_graph(edges)
```

## Related docs

- [Louie Knowledge Graphs](louie-knowledge-graphs.md)
- [Telemetry and Observability](telemetry-observability.md)
- [Graphistry API](../reference/graphistry-api.md)
