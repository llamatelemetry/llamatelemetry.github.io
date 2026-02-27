# Louie Knowledge Graph Guide

`llamatelemetry.louie` bridges LLM output with graph-oriented analysis patterns.

## Package exports

- Client utilities:
  - `LouieClient`
  - `natural_query`
  - `extract_entities`
  - `extract_relationships`
- Knowledge utilities:
  - `KnowledgeExtractor`
  - `build_knowledge_graph`
  - `EntityType`, `RelationType`

## Basic extraction flow

```python
from llamatelemetry.louie import extract_entities, extract_relationships

text = "NVIDIA builds GPUs and CUDA powers accelerated computing."
entities = extract_entities(text)
relations = extract_relationships(text)
```

## Build graph object

```python
from llamatelemetry.louie import build_knowledge_graph

kg = build_knowledge_graph(text)
```

## Natural language query pattern

```python
from llamatelemetry.louie import natural_query

result = natural_query("Find relationships between CUDA and LLM inference.")
```

## Integration recommendations

- Pair with `graphistry` module for visualization.
- Use in notebook pipelines for document-to-graph workflows.
- Keep extraction prompts/domain-specific schemas versioned.
