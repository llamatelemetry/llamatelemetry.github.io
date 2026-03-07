# Louie API Reference

`llamatelemetry.louie` provides AI-powered knowledge extraction and natural language graph analysis.
It combines llamatelemetry LLM inference with GPU-accelerated Graphistry visualization, enabling
natural language queries that produce structured knowledge graphs from unstructured text.

```python
from llamatelemetry.louie import (
    LouieClient, natural_query, extract_entities, extract_relationships,
    KnowledgeExtractor, build_knowledge_graph,
    EntityType, RelationType,
)
```

---

## LouieClient

Client for natural language graph analysis. Combines LLM inference with optional Graphistry
visualization for GPU-accelerated knowledge graph exploration.

### LouieClient(model, server_url, use_local_llm, graphistry_username, graphistry_password)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `Optional[str]` | `None` | Model name for local inference (default: `"gemma-3-1b-Q4_K_M"`) |
| `server_url` | `Optional[str]` | `None` | llama-server URL (default: `"http://localhost:8080"`) |
| `use_local_llm` | `bool` | `True` | Use local llamatelemetry inference vs. server mode |
| `graphistry_username` | `Optional[str]` | `None` | Graphistry Hub username for visualization |
| `graphistry_password` | `Optional[str]` | `None` | Graphistry Hub password |

The LLM backend is lazily initialized on first query. If `use_local_llm` is `True`, creates an
`InferenceEngine` and loads the model. Otherwise, creates a `LlamaCppClient` connected to `server_url`.

### LouieClient.query()

```python
def query(
    self,
    question: str,
    context: Optional[str] = None,
    extract_graph: bool = True,
    max_tokens: int = 1000,
) -> QueryResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | `str` | -- | Natural language question |
| `context` | `Optional[str]` | `None` | Text or data to analyze |
| `extract_graph` | `bool` | `True` | Try to extract and build a Graphistry graph from the response |
| `max_tokens` | `int` | `1000` | Maximum response tokens |

**Returns:** `QueryResult` containing text response, extracted entities, relationships, and optional Graphistry graph object.

The method builds a structured prompt requesting JSON output with `entities` and `relationships` arrays, sends it to the LLM, parses the JSON response, and optionally creates a GPU-accelerated Graphistry knowledge graph on GPU 1.

```python
client = LouieClient()
result = client.query(
    "Extract entities and relationships",
    context="Python is used for AI. TensorFlow is a Python library."
)
print(result.entities)
# [{"id": "Python", "type": "language"}, {"id": "TensorFlow", "type": "technology"}, ...]
print(result.relationships)
# [{"source": "Python", "target": "AI", "type": "used_for"}, ...]
if result.graph:
    result.graph.plot()
```

### LouieClient.extract()

```python
def extract(self, text: str, **kwargs) -> QueryResult
```

Shorthand for extraction. Equivalent to `query("Extract all entities and relationships from this text.", context=text, **kwargs)`.

---

## QueryResult

Dataclass returned by `LouieClient.query()` and related functions.

```python
@dataclass
class QueryResult:
    text: str                              # LLM response text
    entities: List[Dict[str, Any]]         # Extracted entities
    relationships: List[Dict[str, Any]]    # Extracted relationships
    raw_response: str                      # Raw LLM output
    graph: Optional[Any] = None            # Graphistry graph object (if available)
```

---

## Convenience Functions

### natural_query()

```python
def natural_query(
    question: str,
    context: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> QueryResult
```

Quick natural language query. Creates a `LouieClient` internally.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | `str` | -- | Natural language question |
| `context` | `Optional[str]` | `None` | Context text |
| `model` | `Optional[str]` | `None` | Model name (default: `"gemma-3-1b-Q4_K_M"`) |

```python
from llamatelemetry.louie import natural_query
result = natural_query(
    "What are the main technologies?",
    context="Python and TensorFlow are used for machine learning."
)
print(result.text)
```

### extract_entities()

```python
def extract_entities(
    text: str,
    model: Optional[str] = None,
    **kwargs,
) -> List[Dict[str, Any]]
```

Extracts entities from text. Each entity is a dict with `id`, `type`, and `properties` keys.

**Returns:** List of entity dicts.

```python
from llamatelemetry.louie import extract_entities
entities = extract_entities("Python is a programming language.")
# [{"id": "Python", "type": "language", "properties": {}}]
```

### extract_relationships()

```python
def extract_relationships(
    text: str,
    model: Optional[str] = None,
    **kwargs,
) -> List[Dict[str, Any]]
```

Extracts relationships from text. Each relationship is a dict with `source`, `target`, `type`, and `weight` keys.

**Returns:** List of relationship dicts.

```python
from llamatelemetry.louie import extract_relationships
rels = extract_relationships("Python is used for AI development.")
# [{"source": "Python", "target": "AI", "type": "used_for"}]
```

---

## KnowledgeExtractor

Structured knowledge extraction using LLM inference with typed entities and relationships.

### KnowledgeExtractor(model, server_url, entity_types, relationship_types)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `Optional[str]` | `None` | Model name (default: `"gemma-3-1b-Q4_K_M"`) |
| `server_url` | `Optional[str]` | `None` | llama-server URL (uses local engine if `None`) |
| `entity_types` | `Optional[List[EntityType]]` | `None` | Entity types to extract (all types if `None`) |
| `relationship_types` | `Optional[List[RelationType]]` | `None` | Relationship types to extract (all types if `None`) |

### KnowledgeExtractor.extract()

```python
def extract(
    self,
    text: str,
    max_entities: int = 50,
    max_relationships: int = 100,
    max_tokens: int = 2000,
) -> KnowledgeGraph
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | -- | Text to analyze |
| `max_entities` | `int` | `50` | Maximum entities to extract |
| `max_relationships` | `int` | `100` | Maximum relationships to extract |
| `max_tokens` | `int` | `2000` | Maximum LLM response tokens |

**Returns:** `KnowledgeGraph` with typed `Entity` and `Relationship` objects.

```python
extractor = KnowledgeExtractor(model="gemma-3-1b-Q4_K_M")
kg = extractor.extract("""
    Python is a programming language created by Guido van Rossum.
    It is widely used for machine learning with libraries like TensorFlow.
""")
print(f"Entities: {len(kg.entities)}")
print(f"Relationships: {len(kg.relationships)}")
g = kg.to_graphistry()
g.plot()
```

---

## build_knowledge_graph()

```python
def build_knowledge_graph(
    text: str,
    model: Optional[str] = None,
    **kwargs,
) -> KnowledgeGraph
```

Quick helper that creates a `KnowledgeExtractor` and extracts a knowledge graph from text.

```python
from llamatelemetry.louie import build_knowledge_graph
kg = build_knowledge_graph("Python is used for data science. TensorFlow is popular.")
g = kg.to_graphistry()
g.plot()
```

---

## EntityType

Standard entity types for knowledge extraction.

```python
class EntityType(Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    TECHNOLOGY = "technology"
    LANGUAGE = "language"
    PRODUCT = "product"
    EVENT = "event"
    DATE = "date"
    NUMBER = "number"
    OTHER = "other"
```

---

## RelationType

Standard relationship types for knowledge graphs.

```python
class RelationType(Enum):
    USES = "uses"
    CREATES = "creates"
    BELONGS_TO = "belongs_to"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    LOCATED_IN = "located_in"
    WORKS_FOR = "works_for"
    DEPENDS_ON = "depends_on"
    IMPLEMENTS = "implements"
    EXTENDS = "extends"
    CONTAINS = "contains"
    OTHER = "other"
```

---

## Entity

Extracted entity dataclass.

```python
@dataclass
class Entity:
    id: str                              # Entity identifier
    type: EntityType                     # Entity type enum
    properties: Dict[str, Any]           # Additional properties
```

### Entity.to_dict()

```python
def to_dict(self) -> Dict[str, Any]
```

**Returns:** Dict with `id`, `type` (as string), and `properties`.

---

## Relationship

Extracted relationship dataclass.

```python
@dataclass
class Relationship:
    source: str                          # Source entity ID
    target: str                          # Target entity ID
    type: RelationType                   # Relationship type enum
    weight: float = 1.0                  # Relationship strength (0.0 to 1.0)
    properties: Dict[str, Any]           # Additional properties
```

### Relationship.to_dict()

```python
def to_dict(self) -> Dict[str, Any]
```

**Returns:** Dict with `source`, `target`, `type` (as string), `weight`, and `properties`.

---

## KnowledgeGraph

Knowledge graph containing entities and relationships with conversion methods.

```python
@dataclass
class KnowledgeGraph:
    entities: List[Entity]
    relationships: List[Relationship]
    metadata: Dict[str, Any]             # source_text_length, entity_count, relationship_count
```

### KnowledgeGraph.to_dict()

```python
def to_dict(self) -> Dict[str, Any]
```

**Returns:** Dict with `entities`, `relationships` (as dicts), and `metadata`.

### KnowledgeGraph.to_graphistry()

```python
def to_graphistry(self, gpu_id: int = 1) -> Any
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpu_id` | `int` | `1` | GPU for graph operations |

Converts the knowledge graph to a Graphistry graph object using `GraphWorkload.create_knowledge_graph()`. The graph is styled with nodes colored by entity type and edges colored by weight.

**Returns:** Graphistry plotter object (call `.plot()` to render).

```python
kg = build_knowledge_graph("NVIDIA builds GPUs. CUDA runs on NVIDIA GPUs.")
graph = kg.to_graphistry(gpu_id=1)
graph.plot()  # Opens GPU-accelerated visualization
```

---

## End-to-End Example

```python
from llamatelemetry.louie import KnowledgeExtractor, EntityType, RelationType

# Extract with specific types
extractor = KnowledgeExtractor(
    entity_types=[EntityType.TECHNOLOGY, EntityType.LANGUAGE, EntityType.CONCEPT],
    relationship_types=[RelationType.USES, RelationType.IMPLEMENTS, RelationType.DEPENDS_ON],
)

kg = extractor.extract("""
    llamatelemetry is a Python SDK built on OpenTelemetry.
    It uses CUDA for GPU acceleration and llama.cpp for inference.
    The SDK supports GGUF models quantized with NF4 and Q4_K_M.
""")

# Inspect results
for entity in kg.entities:
    print(f"  {entity.type.value}: {entity.id}")

for rel in kg.relationships:
    print(f"  {rel.source} --{rel.type.value}--> {rel.target} (weight={rel.weight})")

# Visualize
g = kg.to_graphistry()
g.plot()
```

---

## Related Documentation

- [Graphistry API](graphistry-api.md) -- GPU graph visualization
- [Core API](core-api.md) -- InferenceEngine for LLM inference
- [Kaggle API](kaggle-api.md) -- Split-GPU setup
- [Louie Knowledge Graphs Guide](../guides/louie-knowledge-graphs.md)
