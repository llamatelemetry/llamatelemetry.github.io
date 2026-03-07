# Kaggle API Reference

`llamatelemetry.kaggle` provides zero-boilerplate utilities for Kaggle notebooks with dual Tesla T4 GPUs.
This module replaces 50+ lines of setup code with a single call, handling GPU detection, secrets loading,
preset selection, and GPU isolation for split LLM/RAPIDS workflows.

```python
from llamatelemetry.kaggle import (
    KaggleEnvironment, quick_setup,
    ServerPreset, TensorSplitMode, PresetConfig, get_preset_config,
    KaggleSecrets, auto_load_secrets, setup_huggingface_auth, setup_graphistry_auth,
    GPUContext, rapids_gpu, llm_gpu, single_gpu, split_gpu_session,
    KagglePipelineConfig, load_grafana_otlp_env_from_kaggle,
    start_server_from_preset, setup_otel_and_client,
)
```

---

## KaggleEnvironment

The main entry point for Kaggle setup. Automatically detects GPUs, loads secrets,
selects optimal presets, and provides factory methods for engine creation.

```python
class KaggleEnvironment:
    gpu_count: int
    gpu_names: List[str]
    total_vram_gb: float
    vram_per_gpu_gb: List[float]
    compute_capability: str
    cuda_version: str
    hf_token: Optional[str]
    graphistry_key_id: Optional[str]
    graphistry_key_secret: Optional[str]
    preset: Optional[ServerPreset]
    telemetry_enabled: bool
    graphistry_enabled: bool
    rapids_gpu_id: int
    llm_gpu_ids: List[int]
```

### KaggleEnvironment.setup()

```python
@classmethod
def setup(
    cls,
    enable_telemetry: bool = True,
    enable_graphistry: bool = False,
    auto_load_secrets: bool = True,
    split_gpu_mode: bool = True,
    verbose: bool = True,
) -> KaggleEnvironment
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_telemetry` | `bool` | `True` | Enable OpenTelemetry tracing |
| `enable_graphistry` | `bool` | `False` | Register Graphistry from secrets |
| `auto_load_secrets` | `bool` | `True` | Load HF_TOKEN, Graphistry keys from Kaggle secrets |
| `split_gpu_mode` | `bool` | `True` | Use GPU 0 for LLM, GPU 1 for RAPIDS |
| `verbose` | `bool` | `True` | Print setup summary |

**Returns:** Configured `KaggleEnvironment` instance.

```python
env = KaggleEnvironment.setup()
print(f"GPUs: {env.gpu_count}x {env.gpu_names[0]}")
print(f"Total VRAM: {env.total_vram_gb:.1f} GB")
print(f"Preset: {env.preset.name}")
```

### KaggleEnvironment.create_engine()

```python
def create_engine(
    self,
    model_name_or_path: str,
    preset: Optional[ServerPreset] = None,
    auto_start: bool = True,
    verbose: bool = True,
    **kwargs,
) -> InferenceEngine
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | `str` | -- | Model name from registry or path to GGUF file |
| `preset` | `Optional[ServerPreset]` | `None` | Override auto-detected preset |
| `auto_start` | `bool` | `True` | Start llama-server automatically |
| `verbose` | `bool` | `True` | Print status messages |
| `**kwargs` | -- | -- | Additional `InferenceEngine.load_model()` parameters |

**Returns:** Configured `InferenceEngine` ready for inference.

```python
env = KaggleEnvironment.setup()
engine = env.create_engine("gemma-3-4b-Q4_K_M")
result = engine.infer("Hello, world!")
print(result.text)
```

### KaggleEnvironment.rapids_context()

```python
def rapids_context(self) -> GPUContext
```

Returns a `GPUContext` configured for RAPIDS operations on the designated RAPIDS GPU (default: GPU 1).

```python
with env.rapids_context():
    import cudf, cugraph
    df = cudf.DataFrame({"x": [1, 2, 3]})
    # All RAPIDS ops on GPU 1
```

### KaggleEnvironment.llm_context()

```python
def llm_context(self) -> GPUContext
```

Returns a `GPUContext` configured for LLM inference GPUs.

### KaggleEnvironment.download_model()

```python
def download_model(
    self,
    repo_id: str,
    filename: str,
    local_dir: Optional[Path] = None,
) -> Path
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repo_id` | `str` | -- | HuggingFace repo ID (e.g., `"unsloth/gemma-3-4b-it-GGUF"`) |
| `filename` | `str` | -- | Model filename (e.g., `"gemma-3-4b-it-Q4_K_M.gguf"`) |
| `local_dir` | `Optional[Path]` | `None` | Download directory (auto-detects Kaggle path if `None`) |

**Returns:** `Path` to the downloaded model file.

### quick_setup()

```python
def quick_setup(**kwargs) -> KaggleEnvironment
```

Alias for `KaggleEnvironment.setup()`. Accepts the same parameters.

---

## ServerPreset

Pre-configured server settings for common GPU environments.

```python
class ServerPreset(Enum):
    AUTO             # Auto-detect optimal settings
    KAGGLE_DUAL_T4   # 2x T4, 30GB total, split-GPU
    KAGGLE_SINGLE_T4 # 1x T4, 15GB
    COLAB_T4         # Google Colab T4
    COLAB_A100       # Google Colab A100
    LOCAL_3090       # RTX 3090 24GB
    LOCAL_4090       # RTX 4090 24GB
    CPU_ONLY         # No GPU
```

### Preset Defaults

| Preset | ctx_size | batch_size | n_parallel | tensor_split | flash_attention |
|--------|----------|------------|------------|--------------|-----------------|
| `KAGGLE_DUAL_T4` | 8192 | 2048 | 4 | `[0.5, 0.5]` | `True` |
| `KAGGLE_SINGLE_T4` | 4096 | 1024 | 2 | `None` | `True` |
| `COLAB_A100` | 16384 | 4096 | 8 | `None` | `True` |
| `LOCAL_4090` | 16384 | 4096 | 8 | `None` | `True` |
| `CPU_ONLY` | 2048 | 512 | 1 | `None` | `False` |

---

## TensorSplitMode

Type-safe GPU tensor split modes, replacing error-prone string values.

```python
class TensorSplitMode(Enum):
    NONE        = "none"       # Single GPU
    EQUAL       = "equal"      # Split equally across all GPUs
    BALANCED    = "balanced"   # Auto-balance based on VRAM
    CUSTOM      = "custom"     # User-defined split ratios
    DUAL_50_50  = "0.5,0.5"
    DUAL_60_40  = "0.6,0.4"
    DUAL_70_30  = "0.7,0.3"
    DUAL_40_60  = "0.4,0.6"
    DUAL_30_70  = "0.3,0.7"
```

### TensorSplitMode.to_string()

```python
def to_string(self) -> Optional[str]
```

**Returns:** Tensor split string for llama-server (e.g., `"0.5,0.5"`), or `None` for `NONE`.

### TensorSplitMode.to_list()

```python
def to_list(self) -> Optional[List[float]]
```

**Returns:** List of floats (e.g., `[0.5, 0.5]`), or `None` for `NONE`.

---

## PresetConfig

Dataclass holding all server configuration values for a preset.

```python
@dataclass
class PresetConfig:
    name: str
    server_url: str = "http://127.0.0.1:8080"
    port: int = 8080
    host: str = "127.0.0.1"
    gpu_layers: int = 99
    tensor_split: Optional[List[float]] = None
    split_mode: TensorSplitMode = TensorSplitMode.NONE
    main_gpu: int = 0
    ctx_size: int = 4096
    batch_size: int = 512
    ubatch_size: int = 128
    flash_attention: bool = True
    use_mmap: bool = True
    n_parallel: int = 1
    enable_metrics: bool = True
    enable_props: bool = True
    enable_slots: bool = True
```

### PresetConfig.to_load_kwargs()

```python
def to_load_kwargs(self) -> Dict[str, Any]
```

**Returns:** Dictionary suitable for `InferenceEngine.load_model(**kwargs)`.

### PresetConfig.to_server_kwargs()

```python
def to_server_kwargs(self) -> Dict[str, Any]
```

**Returns:** Dictionary suitable for `ServerManager.start_server(**kwargs)`, includes `port` and `host`.

---

## get_preset_config()

```python
def get_preset_config(preset: ServerPreset) -> PresetConfig
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `preset` | `ServerPreset` | -- | Preset enum value |

**Returns:** `PresetConfig` for the specified preset. If `AUTO`, auto-detects hardware.

```python
config = get_preset_config(ServerPreset.KAGGLE_DUAL_T4)
print(config.ctx_size)       # 8192
print(config.tensor_split)   # [0.5, 0.5]
print(config.n_parallel)     # 4
```

---

## KaggleSecrets

Wrapper for Kaggle secrets with caching and environment-variable fallback.

```python
class KaggleSecrets:
    KNOWN_SECRETS: List[str]  # HF_TOKEN, OTLP_ENDPOINT, Graphistry keys, etc.
```

### KaggleSecrets(auto_load=True)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_load` | `bool` | `True` | Automatically load all known secrets on init |

### KaggleSecrets.get()

```python
def get(self, name: str, default: Optional[str] = None) -> Optional[str]
```

Retrieves a secret by name. Checks cache, then environment variables, then the Kaggle secrets API.

### KaggleSecrets.set_in_env()

```python
def set_in_env(self, name: str) -> bool
```

Ensures a secret is set in `os.environ`. **Returns:** `True` if found and set.

### KaggleSecrets.list_available()

```python
def list_available(self) -> List[str]
```

**Returns:** List of secret names that have values.

```python
secrets = KaggleSecrets()
hf_token = secrets.get("HF_TOKEN")
print(secrets.list_available())
```

---

## auto_load_secrets()

```python
def auto_load_secrets(
    set_env: bool = True,
    secrets_to_load: Optional[List[str]] = None,
) -> Dict[str, Optional[str]]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `set_env` | `bool` | `True` | Also set secrets in environment variables |
| `secrets_to_load` | `Optional[List[str]]` | `None` | Specific secrets to load (default: all known) |

**Returns:** Dict mapping secret names to values (`None` if not set).

---

## setup_huggingface_auth()

```python
def setup_huggingface_auth() -> bool
```

Loads `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` from secrets and configures `huggingface_hub.login()`.
**Returns:** `True` if authentication was set up.

## setup_graphistry_auth()

```python
def setup_graphistry_auth() -> bool
```

Loads Graphistry credentials from secrets and registers with `pygraphistry`.
**Returns:** `True` if registration succeeded.

---

## GPUContext

Context manager for GPU isolation via `CUDA_VISIBLE_DEVICES`.

```python
class GPUContext:
    gpu_ids: Optional[List[int]]
    restore_on_exit: bool
    visible_devices: str       # property
    is_active: bool            # property
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpu_ids` | `Optional[List[int]]` | `None` | GPU IDs to make visible (`None` = all) |
| `restore_on_exit` | `bool` | `True` | Restore original `CUDA_VISIBLE_DEVICES` on exit |

```python
with GPUContext(gpu_ids=[1]) as ctx:
    import cudf
    df = cudf.DataFrame({"a": [1, 2, 3]})
    print(ctx.visible_devices)  # "1"
# CUDA_VISIBLE_DEVICES restored automatically
```

### GPUContext.set_devices()

```python
def set_devices(self, gpu_ids: List[int]) -> None
```

Change visible devices while inside the context.

---

## GPU Context Helpers

### rapids_gpu()

```python
@contextmanager
def rapids_gpu(gpu_id: int = 1) -> GPUContext
```

Context manager isolating RAPIDS operations to a single GPU.

### llm_gpu()

```python
@contextmanager
def llm_gpu(gpu_ids: Optional[List[int]] = None) -> GPUContext
```

Context manager for LLM inference GPUs (default: `[0, 1]`).

### single_gpu()

```python
@contextmanager
def single_gpu(gpu_id: int = 0) -> GPUContext
```

Context manager for single-GPU operations.

### split_gpu_session()

```python
@contextmanager
def split_gpu_session(llm_gpu: int = 0, graph_gpu: int = 1) -> Dict[str, Any]
```

Yields a dict with `llm_server_kwargs`, `graph_gpu`, and `context` for split-GPU workflows.

```python
with split_gpu_session(llm_gpu=0, graph_gpu=1) as session:
    server_kwargs = session["llm_server_kwargs"]
    # {"main_gpu": 0, "tensor_split": "1.0,0.0"}
```

### get_current_gpu_context()

```python
def get_current_gpu_context() -> Optional[List[int]]
```

**Returns:** Current `CUDA_VISIBLE_DEVICES` as list of GPU IDs, or `None` if not set.

### set_gpu_for_rapids()

```python
def set_gpu_for_rapids(gpu_id: int = 1) -> str
```

Persistent (non-context-manager) GPU assignment. **Returns:** Previous `CUDA_VISIBLE_DEVICES` value.

### reset_gpu_context()

```python
def reset_gpu_context(value: Optional[str] = None) -> None
```

Reset `CUDA_VISIBLE_DEVICES` to a specific value, or unset if `None`.

---

## KagglePipelineConfig

Dataclass for end-to-end pipeline configuration.

```python
@dataclass
class KagglePipelineConfig:
    service_name: str = "llamatelemetry"
    service_version: str = "0.1.1"
    otlp_endpoint: Optional[str] = None
    enable_graphistry: bool = False
    graphistry_server: Optional[str] = None
    enable_llama_metrics: bool = True
    llama_metrics_interval: float = 5.0
```

---

## load_grafana_otlp_env_from_kaggle()

```python
def load_grafana_otlp_env_from_kaggle(
    *,
    endpoint_secret: str = "GRAFANA_OTLP_ENDPOINT",
    headers_secret: str = "GRAFANA_OTLP_HEADERS",
    token_secret: str = "GRAFANA_OTLP_TOKEN",
) -> Dict[str, str]
```

Loads OTLP endpoint and headers from Kaggle secrets into `OTEL_EXPORTER_OTLP_ENDPOINT` and
`OTEL_EXPORTER_OTLP_HEADERS` environment variables. Falls back to `OTLP_ENDPOINT`/`OTLP_TOKEN`
if Grafana-specific secrets are not set.

**Returns:** Dict with `"endpoint"` and `"headers"` keys.

---

## start_server_from_preset()

```python
def start_server_from_preset(
    model_path: str,
    preset: ServerPreset,
    *,
    extra_args: Optional[Dict[str, Any]] = None,
) -> ServerManager
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | -- | Path to GGUF model file |
| `preset` | `ServerPreset` | -- | Server preset enum value |
| `extra_args` | `Optional[Dict[str, Any]]` | `None` | Additional server arguments to merge |

**Returns:** Running `ServerManager` instance (waits up to 180s for readiness).

```python
mgr = start_server_from_preset("/path/to/model.gguf", ServerPreset.KAGGLE_DUAL_T4)
# Server is running and ready at http://127.0.0.1:8080
```

---

## setup_otel_and_client()

```python
def setup_otel_and_client(
    base_url: str,
    cfg: KagglePipelineConfig,
) -> Dict[str, Any]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `str` | -- | llama-server base URL (e.g., `"http://127.0.0.1:8080"`) |
| `cfg` | `KagglePipelineConfig` | -- | Pipeline configuration |

**Returns:** Dict with keys `"tracer"`, `"meter"`, `"client"` (InstrumentedLlamaCppClient), `"gpu_metrics"`.

```python
cfg = KagglePipelineConfig(otlp_endpoint="https://otlp.grafana.net/otlp")
resources = setup_otel_and_client("http://127.0.0.1:8080", cfg)
client = resources["client"]
response = client.chat_completion(messages=[{"role": "user", "content": "Hello"}])
```

---

## Related Documentation

- [Core API Reference](core-api.md) -- InferenceEngine, InferResult
- [Server & Models](server-models.md) -- ServerManager lifecycle
- [Telemetry API](telemetry-api.md) -- OpenTelemetry setup
- [Graphistry API](graphistry-api.md) -- GPU graph visualization
- [Kaggle Quickstart](../get-started/kaggle-quickstart.md) -- Step-by-step guide
