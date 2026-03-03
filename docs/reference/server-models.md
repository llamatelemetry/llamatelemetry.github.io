# Server and Models API Reference

This page documents the server lifecycle management and model discovery systems in llamatelemetry. `ServerManager` handles finding, starting, and stopping the llama-server process. The models module provides smart model loading from registries, HuggingFace, and local paths.

**Modules:** `llamatelemetry.server`, `llamatelemetry.models`

---

## ServerManager

Manages the llama-server process lifecycle: locating the binary, starting with appropriate parameters, health checking, and graceful shutdown. Supports context manager protocol.

### Constructor

```python
class ServerManager:
    def __init__(self, server_url: str = "http://127.0.0.1:8090")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `server_url` | `str` | `"http://127.0.0.1:8090"` | URL where the server will be accessible |

### Context Manager

```python
with ServerManager() as manager:
    manager.start_server(model_path="/path/to/model.gguf", gpu_layers=99)
    # Server is running...
# Server is automatically stopped on exit
```

---

### find_llama_server

```python
def find_llama_server(self) -> Optional[Path]
```

Locate the llama-server executable. Searches in this order:

1. `LLAMA_SERVER_PATH` environment variable
2. Package bootstrap directory (`llamatelemetry/binaries/`)
3. `LLAMA_CPP_DIR` environment variable
4. Cache directory (`~/.cache/llamatelemetry/`)
5. Repository development directories
6. System `PATH`
7. Downloads a fresh binary bundle as last resort

**Returns:** `Path` to the executable, or `None` if not found.

---

### start_server

```python
def start_server(
    self,
    model_path: str,
    port: int = 8090,
    host: str = "127.0.0.1",
    gpu_layers: int = 99,
    ctx_size: int = 2048,
    n_parallel: int = 1,
    batch_size: int = 512,
    ubatch_size: int = 128,
    timeout: int = 60,
    verbose: bool = True,
    skip_gpu_check: bool = False,
    silent: bool = False,
    multi_gpu_config: Optional[Any] = None,
    nccl_config: Optional[Any] = None,
    enable_metrics: bool = False,
    enable_props: bool = False,
    enable_slots: bool = True,
    **kwargs,
) -> bool
```

Start llama-server with the specified configuration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | *required* | Path to GGUF model file |
| `port` | `int` | `8090` | Server port |
| `host` | `str` | `"127.0.0.1"` | Server bind address |
| `gpu_layers` | `int` | `99` | Number of layers to offload to GPU |
| `ctx_size` | `int` | `2048` | Context window size |
| `n_parallel` | `int` | `1` | Number of parallel sequences |
| `batch_size` | `int` | `512` | Logical maximum batch size |
| `ubatch_size` | `int` | `128` | Physical maximum batch size |
| `timeout` | `int` | `60` | Max seconds to wait for server startup |
| `verbose` | `bool` | `True` | Print status messages |
| `skip_gpu_check` | `bool` | `False` | Skip GPU compatibility check |
| `silent` | `bool` | `False` | Suppress all server output |
| `multi_gpu_config` | `Optional[MultiGPUConfig]` | `None` | Multi-GPU configuration object |
| `nccl_config` | `Optional[NCCLConfig]` | `None` | NCCL configuration for environment setup |
| `enable_metrics` | `bool` | `False` | Enable `/metrics` Prometheus endpoint |
| `enable_props` | `bool` | `False` | Enable `/props` endpoint |
| `enable_slots` | `bool` | `True` | Enable `/slots` endpoint |
| `**kwargs` | | | Additional server arguments (see below) |

**Additional kwargs** are mapped to llama-server CLI flags: `flash_attn`, `main_gpu`, `split_mode`, `tensor_split`, `no_mmap`, `mlock`, `threads`, `cache_type_k`, `cache_type_v`, `lora`, `rope_scaling`, and 60+ more parameters.

**Boolean toggle kwargs** support true/false pairs: `perf`, `escape`, `op_offload`, `kv_offload`, `repack`, `warmup`, `context_shift`, `cont_batching`, `webui`.

**Returns:** `True` if the server started successfully.

**Raises:** `FileNotFoundError`, `RuntimeError`

```python
manager = ServerManager(server_url="http://127.0.0.1:8090")
manager.start_server(
    model_path="/models/gemma-3-1b-it-Q4_K_M.gguf",
    gpu_layers=99,
    ctx_size=4096,
    flash_attn=True,
    enable_metrics=True,
)
```

---

### stop_server

```python
def stop_server(self, timeout: float = 10.0) -> bool
```

Stop the running llama-server. Sends `SIGTERM` first, then `SIGKILL` if the graceful shutdown times out.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | `float` | `10.0` | Max seconds to wait for graceful shutdown |

**Returns:** `True` if the server stopped successfully.

---

### Health and Status Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `check_server_health(timeout=2.0)` | `-> bool` | Check if the server responds to `/health` |
| `get_health(timeout=2.0)` | `-> Dict[str, Any]` | Get detailed health JSON (returns `{"status": "loading"}` on 503, `{"status": "unavailable"}` on error) |
| `wait_ready(timeout=60.0, interval=1.0)` | `-> bool` | Block until the server is healthy or timeout |
| `get_metrics(timeout=5.0)` | `-> Optional[str]` | Fetch Prometheus metrics text from `/metrics` |
| `get_models(timeout=5.0)` | `-> Any` | Fetch loaded models from `/v1/models` (OpenAI-compatible) |
| `get_slots(timeout=5.0)` | `-> Any` | Fetch slot status from `/slots` |
| `get_props(timeout=5.0)` | `-> Any` | Fetch server properties from `/props` |
| `get_server_info()` | `-> Dict[str, Any]` | Get local server info: `running`, `url`, `process_id`, `executable` |

```python
manager = ServerManager()
manager.start_server("/path/to/model.gguf", enable_metrics=True)

# Check health
print(manager.check_server_health())  # True

# Get Prometheus metrics
metrics_text = manager.get_metrics()

# Get slot status
slots = manager.get_slots()

# Get loaded model info
models = manager.get_models()
```

---

### restart_server

```python
def restart_server(self, model_path: str, **kwargs) -> bool
```

Stop the current server and restart with new configuration. Accepts the same parameters as `start_server`.

---

### start_from_preset

```python
def start_from_preset(self, model_path: str, preset_name: Any = None, **kwargs) -> bool
```

Start llama-server using a Kaggle preset configuration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | *required* | Path to GGUF model |
| `preset_name` | `str` or `ServerPreset` | `AUTO` | Preset name (`"FAST"`, `"QUALITY"`, `"AUTO"`) |
| `**kwargs` | | | Override values for the preset |

---

## ModelInfo

GGUF model information extractor. Reads metadata from GGUF file headers to provide model information and recommend inference settings.

### Constructor

```python
class ModelInfo:
    def __init__(self, filepath: str)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | `str` | *required* | Path to GGUF model file |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `filepath` | `Path` | Path to the model file |
| `file_size_mb` | `float` | File size in megabytes |
| `architecture` | `Optional[str]` | Model architecture (e.g., `"llama"`, `"gemma"`) |
| `parameter_count` | `Optional[int]` | Estimated parameter count |
| `context_length` | `Optional[int]` | Maximum context length |
| `embedding_length` | `Optional[int]` | Embedding dimension |
| `quantization` | `Optional[str]` | Quantization type |
| `metadata` | `Dict[str, Any]` | Raw GGUF metadata dictionary |

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `from_file(filepath)` | `@classmethod -> ModelInfo` | Create from file path |
| `get_recommended_settings(vram_gb=8.0)` | `-> Dict[str, Any]` | Get recommended `gpu_layers`, `ctx_size`, `batch_size`, `ubatch_size` for given VRAM |
| `to_dict()` | `-> Dict[str, Any]` | Convert to serializable dictionary |

```python
from llamatelemetry.models import ModelInfo
info = ModelInfo.from_file("/models/gemma-3-1b-it-Q4_K_M.gguf")
print(f"Architecture: {info.architecture}")
print(f"Size: {info.file_size_mb:.1f} MB")
settings = info.get_recommended_settings(vram_gb=15.0)
print(f"Recommended GPU layers: {settings['gpu_layers']}")
print(f"Recommended context: {settings['ctx_size']}")
```

---

## ModelManager

Manages a collection of models with scanning, filtering, and selection.

### Constructor

```python
class ModelManager:
    def __init__(self, directories: Optional[List[str]] = None)
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `scan_directories(directories)` | `-> None` | Scan directories for `*.gguf` files recursively |
| `find_by_size(min_mb=0, max_mb=inf)` | `-> List[ModelInfo]` | Filter models by file size range |
| `find_by_architecture(architecture)` | `-> List[ModelInfo]` | Filter models by architecture name (case-insensitive) |
| `get_best_for_vram(vram_gb)` | `-> Optional[ModelInfo]` | Get the largest model that fits in the given VRAM (70% safety margin) |

```python
from llamatelemetry.models import ModelManager
manager = ModelManager(directories=["/kaggle/working/models"])
models = manager.find_by_size(max_mb=4000)  # Models under 4GB
best = manager.get_best_for_vram(vram_gb=15.0)
if best:
    print(f"Best model: {best.filepath.name} ({best.file_size_mb:.0f} MB)")
```

---

## Module-Level Functions

### load_model_smart

```python
def load_model_smart(
    model_name_or_path: str,
    cache_dir: Optional[Path] = None,
    interactive: bool = True,
    force_download: bool = False,
) -> Path
```

Smart model loader with auto-download and confirmation. Handles three cases:

1. **Local path exists** -- returns the path directly
2. **Registry name** (e.g., `"gemma-3-1b-Q4_K_M"`) -- downloads from HuggingFace with optional confirmation
3. **HuggingFace syntax** (`"repo/id:filename.gguf"`) -- downloads directly

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | `str` | *required* | Model name, local path, or `"repo:file"` syntax |
| `cache_dir` | `Optional[Path]` | `None` | Cache directory (default: `llamatelemetry/models/`) |
| `interactive` | `bool` | `True` | Ask for download confirmation |
| `force_download` | `bool` | `False` | Re-download even if cached |

**Returns:** `Path` to the model file, or `None` if the user cancelled.

**Raises:** `ValueError`, `FileNotFoundError`, `ImportError`, `RuntimeError`

```python
from llamatelemetry.models import load_model_smart

# From registry
path = load_model_smart("gemma-3-1b-Q4_K_M")

# Local path
path = load_model_smart("/path/to/model.gguf")

# HuggingFace syntax
path = load_model_smart("unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf")
```

---

### download_model

```python
def download_model(
    repo_id: str,
    filename: str,
    output_dir: Optional[str] = None,
    show_progress: bool = True,
) -> str
```

Download a GGUF model directly from HuggingFace Hub. Requires `huggingface_hub`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repo_id` | `str` | *required* | HuggingFace repository ID |
| `filename` | `str` | *required* | Filename to download |
| `output_dir` | `Optional[str]` | `None` | Output directory (default: current directory) |
| `show_progress` | `bool` | `True` | Show download progress |

**Returns:** Path to the downloaded model file.

---

### list_models

```python
def list_models(directories: Optional[List[str]] = None) -> List[Dict[str, Any]]
```

List all GGUF models in specified directories. Returns model information dictionaries sorted by file size (descending). Each dictionary contains: `filepath`, `filename`, `file_size_mb`, `architecture`, `parameter_count`, `context_length`, `quantization`, `metadata`.

---

### list_registry_models

```python
def list_registry_models() -> Dict[str, Dict[str, Any]]
```

List all models available in the built-in registry (30+ models). Returns a dictionary mapping model name to info dict with keys: `repo`, `file`, `description`, `size_mb`, `min_vram_gb`.

```python
from llamatelemetry.models import list_registry_models
for name, info in list_registry_models().items():
    print(f"{name}: {info['size_mb']}MB - {info['description']}")
```

---

### get_model_recommendations

```python
def get_model_recommendations(vram_gb: float = 8.0) -> List[Dict[str, str]]
```

Get recommended models based on available VRAM. Returns a list of dictionaries with keys: `name`, `repo`, `file`, `size_gb`, `description`. Recommendations are tiered by VRAM: 24GB+, 12GB+, 6GB+, 3GB+, and under 3GB.

---

## SmartModelDownloader

Smart model downloader with VRAM validation and quantization recommendations.

### Constructor

```python
class SmartModelDownloader:
    def __init__(
        self,
        vram_gb: Optional[float] = None,
        cache_dir: Optional[Path] = None,
        auto_recommend: bool = True,
    )
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vram_gb` | `Optional[float]` | `None` | Available VRAM in GB (auto-detected if `None`) |
| `cache_dir` | `Optional[Path]` | `None` | Model cache directory |
| `auto_recommend` | `bool` | `True` | Suggest alternatives if model too large |

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `validate_model(model_name)` | `-> Dict[str, Any]` | Check if a model fits in VRAM. Returns `fits`, `estimated_vram_gb`, `alternative_models`, `warning` |
| `download(model_name_or_path, force=False, warn_on_large=True, interactive=True)` | `-> Optional[Path]` | Download with VRAM validation and warnings |
| `get_recommendations(max_size_gb=None, min_quality="Q4_K_M")` | `-> List[Dict]` | Get top 10 recommended models for available VRAM |

```python
from llamatelemetry.models import SmartModelDownloader
downloader = SmartModelDownloader(vram_gb=15.0)

result = downloader.validate_model("gemma-3-12b-Q4_K_M")
if not result["fits"]:
    print(f"Try: {result['alternative_models']}")

path = downloader.download("gemma-3-4b-Q4_K_M")
```

---

## Related Modules

- [Core API](core-api.md) -- InferenceEngine uses these internally
- [GGUF API](gguf-api.md) -- Detailed GGUF parsing and analysis
- [Client API](client-api.md) -- Direct server communication
