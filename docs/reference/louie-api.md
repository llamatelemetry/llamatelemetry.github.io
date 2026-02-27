# Louie API Reference

## Module: `llamatelemetry.louie.client`

## Types

- `QueryResult`
- `LouieClient`

## Functions

- `natural_query(...)`
- `extract_entities(...)`
- `extract_relationships(...)`

---

## Module: `llamatelemetry.louie.knowledge`

## Types

- `EntityType`
- `RelationType`
- `Entity`
- `Relationship`
- `KnowledgeGraph`
- `KnowledgeExtractor`

## Function

- `build_knowledge_graph(...)`

## Typical usage

```python
from llamatelemetry.louie import build_knowledge_graph

kg = build_knowledge_graph("NVIDIA creates GPUs and CUDA powers AI workloads.")
print(kg)
```
