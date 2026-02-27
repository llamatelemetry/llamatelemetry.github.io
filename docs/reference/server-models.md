# Server and Models API Reference

## Module: `llamatelemetry.server`

## Class: `ServerManager`

### Methods

- `find_llama_server()`
- `check_server_health(timeout=2.0)`
- `start_server(...)`
- `stop_server(timeout=10.0)`
- `get_server_info()`
- `restart_server(model_path, **kwargs)`

### Important `start_server` parameters

- `model_path`
- `host`, `port`
- `gpu_layers`, `ctx_size`
- `batch_size`, `ubatch_size`
- `n_parallel`
- `silent`
- extra kwargs mapped to server flags (for example `flash_attn`)

---

## Module: `llamatelemetry.models`

## Classes

- `ModelInfo`
- `ModelManager`
- `SmartModelDownloader`

## Functions

- `list_models(directories=None)`
- `download_model(repo_id, filename, output_dir=None)`
- `get_model_recommendations(vram_gb=8.0)`
- `print_model_catalog(vram_gb=None)`
- `load_model_smart(model_name_or_path, ...)`
- `list_registry_models()`
- `print_registry_models(vram_gb=None)`

## `SmartModelDownloader` methods

- `validate_model(model_name)`
- `download(model_name_or_path, force=False, ...)`
- `get_recommendations(max_size_gb=None, min_quality="Q4_K_M")`

## Common pattern

```python
from llamatelemetry.models import SmartModelDownloader

downloader = SmartModelDownloader(vram_gb=15.0)
print(downloader.validate_model("gemma-3-4b-Q4_K_M"))
```
