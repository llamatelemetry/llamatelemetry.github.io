# Model Management Guide

`llamatelemetry.models` provides discovery, metadata extraction, smart download, and recommendations.

## Smart loader

```python
from llamatelemetry.models import load_model_smart

path = load_model_smart("gemma-3-1b-Q4_K_M", interactive=True)
```

Accepted inputs:

- Registry model name
- Local file path
- Hugging Face syntax: `"repo/id:file.gguf"`

## Registry utilities

```python
from llamatelemetry.models import list_registry_models, print_registry_models

models = list_registry_models()
print_registry_models(vram_gb=8.0)
```

## Metadata extraction

```python
from llamatelemetry.models import ModelInfo

info = ModelInfo.from_file("/path/model.gguf")
print(info.architecture, info.context_length, info.file_size_mb)
print(info.get_recommended_settings(vram_gb=15.0))
```

## Smart downloader with VRAM validation

```python
from llamatelemetry.models import SmartModelDownloader

downloader = SmartModelDownloader(vram_gb=15.0)
validation = downloader.validate_model("gemma-3-12b-Q4_K_M")
print(validation)
```

## Directory scanning

```python
from llamatelemetry.models import list_models, ModelManager

all_models = list_models(["/kaggle/working/models"])
manager = ModelManager(["/kaggle/working/models"])
best = manager.get_best_for_vram(vram_gb=8.0)
```

## Recommended workflow

1. Validate fit with `SmartModelDownloader`.
2. Download or reuse cache via `load_model_smart`.
3. Inspect `ModelInfo` for context and settings.
4. Feed tuned settings into `InferenceEngine.load_model`.
