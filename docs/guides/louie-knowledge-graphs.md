# Louie Knowledge Graphs

`llamatelemetry.louie` provides knowledge extraction helpers and a client for natural language graph analysis.

## Components

- `KnowledgeExtractor` — extract entities and relationships from text
- `KnowledgeGraph` — structured graph container
- `LouieClient` — natural language graph queries

## Example

```python
from llamatelemetry.louie.knowledge import build_knowledge_graph

text = "NVIDIA created CUDA and released it in 2007."
kg = build_knowledge_graph(text)
print(kg.entities)
print(kg.relationships)
```

## Related docs

- [Graphistry and RAPIDS](graphistry-rapids.md)
- [Louie API](../reference/louie-api.md)
