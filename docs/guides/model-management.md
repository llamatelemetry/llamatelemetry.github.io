# Model Management

`llamatelemetry.models` provides utilities for GGUF model discovery, registry-based downloads, and metadata inspection.

## Key components

- `ModelInfo`: parse GGUF metadata and recommend settings
- `ModelManager`: store and manage model collections
- `load_model_smart`: registry-aware model loader
- `download_model`: HuggingFace download helper
- Registry helpers: `list_registry_models`, `get_model_recommendations`

## Smart model loading

```python
from llamatelemetry.models import load_model_smart

model_path = load_model_smart("gemma-3-1b-Q4_K_M")
print(model_path)
```

`load_model_smart` handles:

- Registry names
- Local filesystem paths
- HuggingFace repo + filename syntax

## Inspecting GGUF metadata

```python
from llamatelemetry.models import ModelInfo

info = ModelInfo.from_file("/path/to/model.gguf")
print(info.architecture, info.context_length, info.quantization)
print(info.get_recommended_settings(vram_gb=8))
```

## Listing available models

```python
from llamatelemetry.models import list_models

models = list_models()
for m in models:
    print(m["filename"], m["file_size_mb"])
```

## Registry and recommendations

```python
from llamatelemetry.models import list_registry_models, get_model_recommendations

registry = list_registry_models()
recommendations = get_model_recommendations(vram_gb=16)
```

## Related reference

- [Server and Models](../reference/server-models.md)
- [GGUF API](../reference/gguf-api.md)
