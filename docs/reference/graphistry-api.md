# Graphistry API

The `llamatelemetry.graphistry` package provides utilities for connecting to Graphistry and constructing graph visualizations from inference data.

## Key modules

- `connector` — authentication and connection helpers
- `viz` — graph plotting helpers
- `builders` — build graph structures from entities and relationships
- `workload` — workload summaries
- `rapids` — RAPIDS GPU DataFrame helpers

## Example

```python
from llamatelemetry.graphistry.connector import GraphistryConnector

connector = GraphistryConnector()
connector.login()
```

## Related docs

- [Graphistry and RAPIDS Guide](../guides/graphistry-rapids.md)
- [Louie Knowledge Graphs](../guides/louie-knowledge-graphs.md)
