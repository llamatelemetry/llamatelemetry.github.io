# Louie Knowledge Graphs

The `llamatelemetry.louie` module provides AI-powered knowledge extraction and natural language graph analysis. It combines LLM inference with structured entity and relationship extraction, producing knowledge graphs that can be visualized with Graphistry. This guide covers the `LouieClient` for natural language queries, the `KnowledgeExtractor` for structured extraction, entity and relationship types, and integration with GPU-accelerated graph visualization.

## Overview

The Louie integration enables a pipeline from unstructured text to visual knowledge graphs:

1. Feed text to the LLM via `LouieClient` or `KnowledgeExtractor`
2. The LLM extracts entities and relationships as structured JSON
3. Results are parsed into `Entity`, `Relationship`, and `KnowledgeGraph` dataclasses
4. Optionally visualize the graph with Graphistry on a second GPU

```python
from llamatelemetry.louie import build_knowledge_graph

kg = build_knowledge_graph("""
    Python is a programming language created by Guido van Rossum.
    It is widely used for machine learning with libraries like TensorFlow.
""")

print(f"Entities: {len(kg.entities)}")
print(f"Relationships: {len(kg.relationships)}")
```

---

## LouieClient: Natural Language Graph Queries

The `LouieClient` is the primary interface for combining LLM inference with knowledge extraction and optional Graphistry visualization.

### Initialization

```python
from llamatelemetry.louie import LouieClient

# Default: uses local LLM with gemma-3-1b-Q4_K_M
client = LouieClient()

# Custom model and server
client = LouieClient(
    model="gemma-3-1b-Q4_K_M",
    server_url="http://localhost:8080",
    use_local_llm=True,
    graphistry_username="your_username",
    graphistry_password="your_password",
)
```

The client lazily initializes its LLM backend on first use. When `use_local_llm=True`, it creates an `InferenceEngine` and loads the specified model. If that fails, it falls back to a `LlamaCppClient` connecting to the server URL.

### Querying with Context

```python
result = client.query(
    question="What are the main technologies and their relationships?",
    context="""
        NVIDIA created CUDA for GPU computing. TensorFlow and PyTorch
        both use CUDA for acceleration. Google developed TensorFlow,
        while Meta created PyTorch.
    """,
    extract_graph=True,
    max_tokens=1000,
)

# Access the results
print(result.text)            # Natural language answer
print(result.entities)        # List of entity dicts
print(result.relationships)   # List of relationship dicts

# Visualize if Graphistry is configured
if result.graph:
    result.graph.plot()
```

The `QueryResult` dataclass contains:

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Natural language answer from the LLM |
| `entities` | `list[dict]` | Extracted entities with id, type, and properties |
| `relationships` | `list[dict]` | Extracted relationships with source, target, type, weight |
| `raw_response` | `str` | Unprocessed LLM response |
| `graph` | `Any` or `None` | Graphistry graph object (if extraction + Graphistry succeeded) |

### Shorthand Extraction

For simple text-to-knowledge extraction without a specific question:

```python
result = client.extract("""
    Kubernetes orchestrates Docker containers.
    Helm is a package manager for Kubernetes.
""")

for entity in result.entities:
    print(f"  {entity['id']} ({entity['type']})")
```

---

## Convenience Functions

For quick one-off tasks, use the module-level helper functions. Each creates a temporary `LouieClient` internally:

### Natural Language Query

```python
from llamatelemetry.louie import natural_query

result = natural_query(
    "What are the main technologies mentioned?",
    context="Python and TensorFlow are used for machine learning on NVIDIA GPUs.",
    model="gemma-3-1b-Q4_K_M",
)
print(result.text)
```

### Extract Entities Only

```python
from llamatelemetry.louie import extract_entities

entities = extract_entities("Python is a programming language created by Guido van Rossum.")
for e in entities:
    print(f"  {e['id']}: {e['type']}")
# Output:
#   Python: language
#   Guido van Rossum: person
```

### Extract Relationships Only

```python
from llamatelemetry.louie import extract_relationships

rels = extract_relationships("Python is used for AI development. TensorFlow runs on CUDA.")
for r in rels:
    print(f"  {r['source']} --[{r['type']}]--> {r['target']}")
```

---

## Entity and Relationship Types

The `knowledge` module defines standard enums for categorizing extracted data. These guide the LLM's extraction prompt and provide type safety when working with results.

### Entity Types

```python
from llamatelemetry.louie import EntityType

# Available types:
# PERSON, ORGANIZATION, LOCATION, CONCEPT, TECHNOLOGY,
# LANGUAGE, PRODUCT, EVENT, DATE, NUMBER, OTHER
```

| Type | Example Entities |
|---|---|
| `PERSON` | Guido van Rossum, Linus Torvalds |
| `ORGANIZATION` | NVIDIA, Google, Meta |
| `TECHNOLOGY` | CUDA, Docker, Kubernetes |
| `LANGUAGE` | Python, Rust, C++ |
| `PRODUCT` | TensorFlow, PyTorch |
| `CONCEPT` | machine learning, gradient descent |
| `LOCATION` | Silicon Valley, Amsterdam |
| `EVENT` | GTC Conference, PyCon |

### Relationship Types

```python
from llamatelemetry.louie import RelationType

# Available types:
# USES, CREATES, BELONGS_TO, RELATED_TO, PART_OF,
# LOCATED_IN, WORKS_FOR, DEPENDS_ON, IMPLEMENTS,
# EXTENDS, CONTAINS, OTHER
```

---

## KnowledgeExtractor: Structured Extraction

For fine-grained control over the extraction process, including filtering by entity and relationship types, use `KnowledgeExtractor` directly.

### Basic Extraction

```python
from llamatelemetry.louie import KnowledgeExtractor

extractor = KnowledgeExtractor(model="gemma-3-1b-Q4_K_M")

kg = extractor.extract(
    text="""
        NVIDIA developed CUDA in 2006. Tesla GPUs use CUDA for parallel computing.
        cuDNN is a library built on top of CUDA for deep learning.
    """,
    max_entities=50,
    max_relationships=100,
    max_tokens=2000,
)

print(f"Found {len(kg.entities)} entities and {len(kg.relationships)} relationships")
```

### Filtering Entity and Relationship Types

Restrict extraction to specific categories:

```python
from llamatelemetry.louie import KnowledgeExtractor, EntityType, RelationType

extractor = KnowledgeExtractor(
    model="gemma-3-1b-Q4_K_M",
    entity_types=[EntityType.TECHNOLOGY, EntityType.ORGANIZATION, EntityType.PRODUCT],
    relationship_types=[RelationType.CREATES, RelationType.USES, RelationType.DEPENDS_ON],
)

kg = extractor.extract("NVIDIA created CUDA. TensorFlow uses CUDA for GPU acceleration.")
```

### Using a Remote Server

Point the extractor at a running llama-server instead of loading a local model:

```python
extractor = KnowledgeExtractor(server_url="http://localhost:8080")
kg = extractor.extract("Your text here...")
```

---

## Working with KnowledgeGraph Objects

The `KnowledgeGraph` dataclass provides structured access to extraction results and serialization utilities.

### Inspecting Results

```python
from llamatelemetry.louie.knowledge import KnowledgeGraph, Entity, Relationship

# After extraction
for entity in kg.entities:
    print(f"Entity: {entity.id} (type={entity.type.value})")
    if entity.properties:
        print(f"  Properties: {entity.properties}")

for rel in kg.relationships:
    print(f"Relation: {rel.source} --[{rel.type.value}]--> {rel.target} (weight={rel.weight})")
```

### Serialization

```python
# Convert to dictionary (for JSON serialization)
data = kg.to_dict()

# Access metadata
print(kg.metadata)
# {'source_text_length': 142, 'entity_count': 5, 'relationship_count': 3}
```

### Visualizing with Graphistry

Convert a knowledge graph directly to a Graphistry visualization:

```python
# Requires Graphistry configuration
graph = kg.to_graphistry(gpu_id=1)
graph.plot()
```

This uses the split-GPU pattern: LLM inference on GPU 0, graph operations on GPU 1.

---

## End-to-End Example

Here is a complete workflow from text input to visual knowledge graph:

```python
from llamatelemetry.louie import LouieClient

# Initialize with Graphistry credentials
client = LouieClient(
    model="gemma-3-1b-Q4_K_M",
    graphistry_username="your_user",
    graphistry_password="your_pass",
)

# Analyze a technical document
document = """
Apache Spark is an open-source distributed computing framework developed by AMPLab
at UC Berkeley. It provides APIs in Python (PySpark), Java, Scala, and R.
Spark runs on Hadoop YARN, Apache Mesos, Kubernetes, or standalone clusters.
Databricks, founded by the creators of Spark, provides a managed Spark platform.
"""

result = client.query(
    question="Map all technologies and organizations with their relationships",
    context=document,
    extract_graph=True,
)

# Print structured results
print("=== Entities ===")
for e in result.entities:
    print(f"  [{e.get('type', 'unknown')}] {e['id']}")

print("\n=== Relationships ===")
for r in result.relationships:
    print(f"  {r['source']} --{r['type']}--> {r['target']}")

# Visualize
if result.graph:
    result.graph.plot()
```

---

## Best Practices

1. **Provide clear context** -- The LLM extracts better knowledge when the input text is well-structured and focused on a specific domain.
2. **Filter types for precision** -- Use `entity_types` and `relationship_types` on `KnowledgeExtractor` to reduce noise and focus on relevant categories.
3. **Use the split-GPU pattern** -- On dual-GPU setups (like Kaggle T4 x2), run LLM inference on GPU 0 and Graphistry/graph operations on GPU 1.
4. **Handle parse failures gracefully** -- The LLM may not always produce valid JSON. The parser returns empty lists on failure rather than raising exceptions.
5. **Increase `max_tokens` for complex texts** -- Longer texts with many entities need more tokens for the LLM to produce a complete extraction.
6. **Cache results** -- For repeated analysis of the same text, save the `KnowledgeGraph.to_dict()` output rather than re-running extraction.

## Related Reference

- [Louie API Reference](../reference/louie-api.md)
- [Graphistry and RAPIDS Guide](graphistry-rapids.md)
- [Inference Engine Guide](inference-engine.md)
