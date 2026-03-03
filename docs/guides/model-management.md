# Model Management

llamatelemetry provides a complete model management system for discovering, downloading, inspecting, and selecting GGUF models. The system includes a curated registry of 22 models, smart downloading from HuggingFace Hub, GGUF metadata inspection, and VRAM-aware recommendations.

## Overview

The model management layer consists of:

- **MODEL_REGISTRY** -- a built-in catalog of 22 pre-tested GGUF models
- **SmartModelDownloader** -- handles downloads from HuggingFace with caching
- **ModelInfo** -- parses GGUF metadata and recommends GPU settings
- **ModelManager** -- manages local model storage and lookup
- **load_model_smart()** -- unified loader that resolves registry names, local paths, and HuggingFace references

## The Model Registry

The built-in `MODEL_REGISTRY` contains 22 curated models that have been tested with llamatelemetry:

```python
from llamatelemetry._internal.bootstrap import MODEL_REGISTRY

for name, info in MODEL_REGISTRY.items():
    print(f"{name}: {info.get('file_size_mb', '?')} MB")
```

### Registry Model Categories

| Category | Models | Size Range |
|----------|--------|------------|
| **Tiny (< 1B)** | TinyLlama, SmolLM | 500 MB - 1 GB |
| **Small (1-3B)** | Gemma-3-1B, Llama-3.2-1B, Phi-4-mini, Qwen-2.5-1.5B | 1 - 2 GB |
| **Medium (3-8B)** | Llama-3.1-8B, Mistral-7B, Gemma-2-9B | 4 - 6 GB |
| **Large (8B+)** | Llama-3.3-70B, DeepSeek, Mixtral | 6 - 40+ GB |

### Recommended Models for Tesla T4 (16 GB VRAM)

| Model | Quantization | Size | Use Case |
|-------|-------------|------|----------|
| `gemma-3-1b-Q4_K_M` | Q4_K_M | ~0.8 GB | Fast prototyping, testing |
| `llama-3.2-1b-Q4_K_M` | Q4_K_M | ~0.9 GB | General purpose, lightweight |
| `phi-4-mini-Q4_K_M` | Q4_K_M | ~2.3 GB | Coding tasks, reasoning |
| `llama-3.1-8b-Q4_K_M` | Q4_K_M | ~4.9 GB | High-quality general purpose |
| `mistral-7b-Q4_K_M` | Q4_K_M | ~4.4 GB | Instruction following |

## Smart Model Loading

The `load_model_smart()` function is the primary way to resolve and download models:

```python
from llamatelemetry.models import load_model_smart

# From registry -- looks up MODEL_REGISTRY, downloads if needed
model_path = load_model_smart("gemma-3-1b-Q4_K_M")

# From local filesystem
model_path = load_model_smart("/home/user/models/custom-model.gguf")

# From HuggingFace -- repo:filename syntax
model_path = load_model_smart(
    "bartowski/gemma-2-2b-it-GGUF:gemma-2-2b-it-Q4_K_M.gguf"
)
```

### Resolution Order

1. **Local path** -- if the string is a valid filesystem path to an existing `.gguf` file, use it directly
2. **Registry name** -- if it matches a key in `MODEL_REGISTRY`, download from the registered HuggingFace URL
3. **HuggingFace reference** -- if it contains a colon (`repo:filename`), download from HuggingFace Hub
4. **Error** -- raise an exception with suggestions

### Download Caching

Downloaded models are cached in `~/.cache/llamatelemetry/models/`. Subsequent calls to `load_model_smart()` with the same model name return the cached path immediately.

## ModelInfo -- GGUF Metadata Inspection

`ModelInfo` reads GGUF file headers to extract model metadata:

```python
from llamatelemetry.models import ModelInfo

info = ModelInfo.from_file("/path/to/model.gguf")

print(f"Architecture: {info.architecture}")
print(f"Parameters: {info.parameters}")
print(f"Context length: {info.context_length}")
print(f"Quantization: {info.quantization}")
print(f"File size: {info.file_size_mb:.1f} MB")
print(f"Embedding size: {info.embedding_size}")
print(f"Layers: {info.n_layers}")
```

### VRAM Recommendations

```python
recommendations = info.get_recommended_settings(vram_gb=16)
print(f"GPU layers: {recommendations['gpu_layers']}")
print(f"Context size: {recommendations['ctx_size']}")
print(f"Estimated VRAM: {recommendations['estimated_vram_gb']:.1f} GB")
```

The recommendation engine accounts for:

- Model size (parameters and quantization)
- KV cache memory (scales with context size and layers)
- CUDA runtime overhead (~500 MB)
- A safety margin to avoid OOM

## ModelManager -- Collection Management

`ModelManager` provides a higher-level interface for managing multiple models:

```python
from llamatelemetry.models import ModelManager

mm = ModelManager()

# List all locally available models
local_models = mm.list_local_models()
for m in local_models:
    print(f"{m['name']}: {m['path']}")

# Download a model
path = mm.download("gemma-3-1b-Q4_K_M")

# Remove a cached model
mm.remove("gemma-3-1b-Q4_K_M")
```

## SmartModelDownloader

For fine-grained download control:

```python
from llamatelemetry.models import SmartModelDownloader

downloader = SmartModelDownloader()

# Download with progress callback
path = downloader.download(
    repo_id="bartowski/gemma-2-2b-it-GGUF",
    filename="gemma-2-2b-it-Q4_K_M.gguf",
    cache_dir="~/.cache/llamatelemetry/models",
)
```

The downloader uses `huggingface_hub` under the hood and supports:

- Resume of interrupted downloads
- Progress bar display via `tqdm`
- Token-based authentication for gated models
- Automatic cache management

### Gated Model Authentication

Some HuggingFace models require authentication:

```python
import os
os.environ["HF_TOKEN"] = "hf_your_token_here"

# Or set it before downloading
path = load_model_smart("meta-llama/Llama-3.1-8B-GGUF:model-Q4_K_M.gguf")
```

## Listing Available Models

```python
from llamatelemetry.models import list_models, list_registry_models

# List all locally cached models
local = list_models()
for m in local:
    print(f"{m['filename']}: {m['file_size_mb']:.0f} MB")

# List all models in the registry
registry = list_registry_models()
for name in registry:
    print(name)
```

## Model Recommendations

Get recommendations based on available VRAM:

```python
from llamatelemetry.models import get_model_recommendations

# Get models that fit in 16 GB VRAM
recs = get_model_recommendations(vram_gb=16)
for rec in recs:
    print(f"{rec['name']}: {rec['estimated_vram_gb']:.1f} GB VRAM")
```

## Bootstrap System

On first import, llamatelemetry checks for the `llama-server` binary and essential models. The bootstrap process:

1. Downloads the `llama-server` release bundle (~961 MB) if missing
2. Extracts binaries and CUDA shared libraries to `~/.cache/llamatelemetry/`
3. Configures `LD_LIBRARY_PATH` for the bundled CUDA libraries
4. Caches model files in `~/.cache/llamatelemetry/models/`

```python
from llamatelemetry._internal.bootstrap import bootstrap

# Force re-bootstrap
bootstrap(force=True)
```

## Best Practices

- **Start with small models** (1B parameters) for development and testing on T4 GPUs.
- **Use registry names** for reproducibility -- they pin specific quantization variants.
- **Check VRAM before loading** using `ModelInfo.get_recommended_settings()`.
- **Cache models locally** to avoid repeated downloads on Kaggle.
- **Use Q4_K_M quantization** as the default -- it provides the best quality/size tradeoff.
- **Set `HF_TOKEN`** for gated models before calling `load_model_smart()`.

## Complete Example

```python
from llamatelemetry.models import (
    load_model_smart,
    ModelInfo,
    list_registry_models,
    get_model_recommendations,
)

# 1. Check what models fit on our GPU
recs = get_model_recommendations(vram_gb=16)
print("Recommended models for 16 GB VRAM:")
for r in recs[:5]:
    print(f"  {r['name']}")

# 2. List registry models
print("\nRegistry models:")
for name in list_registry_models():
    print(f"  {name}")

# 3. Load a model
model_path = load_model_smart("gemma-3-1b-Q4_K_M")
print(f"\nModel path: {model_path}")

# 4. Inspect metadata
info = ModelInfo.from_file(model_path)
print(f"Architecture: {info.architecture}")
print(f"Quantization: {info.quantization}")
print(f"Context length: {info.context_length}")

# 5. Get optimal settings for our GPU
settings = info.get_recommended_settings(vram_gb=16)
print(f"Recommended GPU layers: {settings['gpu_layers']}")
print(f"Recommended context size: {settings['ctx_size']}")
```

## Related

- [Inference Engine](inference-engine.md) -- uses model management internally
- [Quantization](quantization.md) -- GGUF quantization details
- [GGUF API Reference](../reference/gguf-api.md)
- [Server and Models Reference](../reference/server-models.md)
