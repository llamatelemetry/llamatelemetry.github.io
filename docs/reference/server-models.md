# Server and Models

## ServerManager

Lifecycle manager for `llama-server`.

**Key methods:**

- `find_llama_server()` — locate or download the binary
- `start_server(model_path, **kwargs)` — launch server
- `stop_server()` — terminate server
- `check_server_health()` / `get_health()` — readiness checks
- `get_metrics()` — Prometheus metrics text
- `get_models()` — OpenAI-style model list
- `get_slots()` — slot status

## ModelInfo

Parses GGUF metadata and provides recommended settings.

```python
from llamatelemetry.models import ModelInfo
info = ModelInfo.from_file("model.gguf")
print(info.get_recommended_settings(vram_gb=8))
```

## ModelManager

In-memory manager for model registries and metadata.

## Registry helpers

- `list_registry_models()`
- `print_registry_models()`
- `get_model_recommendations()`

## Smart loading

- `load_model_smart(model_name_or_path, ...)`
- `download_model(repo_id, filename, ...)`

## Related modules

- [Core API](core-api.md)
- [GGUF API](gguf-api.md)
