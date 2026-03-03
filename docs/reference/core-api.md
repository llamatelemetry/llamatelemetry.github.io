# Core API Reference

The core API provides the high-level entry point for LLM inference with llamatelemetry. It includes the `InferenceEngine` class for managing the full inference lifecycle, the `InferResult` data wrapper, and convenience functions for CUDA detection and quick inference.

**Module:** `llamatelemetry`

---

## InferenceEngine

High-level Python interface for LLM inference with CUDA acceleration. Manages server lifecycle, model loading, inference execution, and performance metrics collection. Supports automatic server startup, telemetry integration, and multi-GPU configurations.

### Constructor

```python
class InferenceEngine:
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8090",
        enable_telemetry: bool = False,
        telemetry_config: Optional[Dict[str, Any]] = None,
    )
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `server_url` | `str` | `"http://127.0.0.1:8090"` | URL of the llama-server backend |
| `enable_telemetry` | `bool` | `False` | Enable OpenTelemetry tracing and metrics |
| `telemetry_config` | `Optional[Dict[str, Any]]` | `None` | Telemetry configuration dictionary (keys: `service_name`, `service_version`, `otlp_endpoint`, `enable_graphistry`, `graphistry_server`, `enable_llama_metrics`, `llama_metrics_interval`) |

### Context Manager

`InferenceEngine` supports the context manager protocol for automatic cleanup:

```python
with InferenceEngine() as engine:
    engine.load_model("gemma-3-1b-Q4_K_M")
    result = engine.infer("What is AI?")
    print(result.text)
# Server is automatically stopped on exit
```

---

### load_model

```python
def load_model(
    self,
    model_name_or_path: str,
    gpu_layers: Optional[int] = None,
    ctx_size: Optional[int] = None,
    auto_start: bool = True,
    auto_configure: bool = True,
    n_parallel: int = 1,
    verbose: bool = True,
    interactive_download: bool = True,
    silent: bool = False,
    report_suitability: bool = False,
    **kwargs,
) -> Optional[bool]
```

Load a GGUF model for inference with smart loading and auto-configuration. Supports three loading modes: registry name, local path, or HuggingFace syntax.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | `str` | *required* | Model name from registry (e.g., `"gemma-3-1b-Q4_K_M"`), local file path, or HuggingFace syntax (`"repo/id:filename.gguf"`) |
| `gpu_layers` | `Optional[int]` | `None` | Number of layers to offload to GPU. `None` triggers auto-configuration |
| `ctx_size` | `Optional[int]` | `None` | Context size in tokens. `None` triggers auto-configuration |
| `auto_start` | `bool` | `True` | Automatically start llama-server if not running |
| `auto_configure` | `bool` | `True` | Automatically configure optimal GPU layers, context size, and batch sizes |
| `n_parallel` | `int` | `1` | Number of parallel inference sequences |
| `verbose` | `bool` | `True` | Print status messages during loading |
| `interactive_download` | `bool` | `True` | Ask for user confirmation before downloading models |
| `silent` | `bool` | `False` | Suppress all llama-server output and warnings |
| `report_suitability` | `bool` | `False` | Print GGUF suitability report for Kaggle T4 |
| `**kwargs` | | | Additional server parameters: `batch_size`, `ubatch_size`, `multi_gpu_config`, `nccl_config`, `split_mode`, `main_gpu`, `tensor_split`, `flash_attn`, `enable_metrics`, etc. |

**Returns:** `True` if the model loaded successfully. `None` if the user cancelled an interactive download.

**Raises:**

- `FileNotFoundError` -- Model file not found
- `ConnectionError` -- Server not running and `auto_start=False`
- `RuntimeError` -- Server fails to start

```python
engine = InferenceEngine()

# Auto-download from registry
engine.load_model("gemma-3-1b-Q4_K_M")

# Local path with manual settings
engine.load_model("/path/to/model.gguf", gpu_layers=20, ctx_size=2048)

# HuggingFace download
engine.load_model("unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf")

# Multi-GPU with suitability report
from llamatelemetry.api.multigpu import MultiGPUConfig, SplitMode
config = MultiGPUConfig(split_mode=SplitMode.LAYER, tensor_split=[0.5, 0.5])
engine.load_model("gemma-3-1b-Q4_K_M", multi_gpu_config=config, report_suitability=True)
```

---

### infer

```python
def infer(
    self,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    seed: int = 0,
    stop_sequences: Optional[List[str]] = None,
) -> InferResult
```

Run inference on a single prompt using the native `/completion` endpoint.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | *required* | Input prompt text |
| `max_tokens` | `int` | `128` | Maximum tokens to generate |
| `temperature` | `float` | `0.7` | Sampling temperature (higher = more random) |
| `top_p` | `float` | `0.9` | Nucleus sampling threshold |
| `top_k` | `int` | `40` | Top-k sampling limit |
| `seed` | `int` | `0` | Random seed (0 = random) |
| `stop_sequences` | `Optional[List[str]]` | `None` | List of stop sequences |

**Returns:** `InferResult` object with generated text and metrics.

```python
result = engine.infer("Explain quantum computing", max_tokens=256, temperature=0.5)
if result.success:
    print(result.text)
    print(f"Generated {result.tokens_generated} tokens in {result.latency_ms:.0f}ms")
else:
    print(f"Error: {result.error_message}")
```

---

### generate

```python
def generate(self, prompt: str, **kwargs) -> InferResult
```

Alias for `infer()` to align with common LLM SDK conventions. Accepts the same parameters.

---

### infer_stream

```python
def infer_stream(
    self,
    prompt: str,
    callback: Any,
    max_tokens: int = 128,
    temperature: float = 0.7,
    **kwargs,
) -> InferResult
```

Run streaming inference with a callback for each generated chunk.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | *required* | Input prompt text |
| `callback` | `Callable[[str], None]` | *required* | Function called with each text chunk |
| `max_tokens` | `int` | `128` | Maximum tokens to generate |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `**kwargs` | | | Additional parameters: `top_p`, `top_k`, `seed` |

**Returns:** `InferResult` with the complete response.

---

### batch_infer

```python
def batch_infer(
    self,
    prompts: List[str],
    max_tokens: int = 128,
    **kwargs,
) -> List[InferResult]
```

Run inference on multiple prompts sequentially.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompts` | `List[str]` | *required* | List of input prompts |
| `max_tokens` | `int` | `128` | Maximum tokens per prompt |
| `**kwargs` | | | Additional parameters forwarded to `infer()` |

**Returns:** List of `InferResult` objects, one per prompt.

---

### get_metrics

```python
def get_metrics(self) -> Dict[str, Any]
```

Get current performance metrics accumulated across all inference calls.

**Returns:** Dictionary with two sub-dictionaries:

- `latency` -- `mean_ms`, `p50_ms`, `p95_ms`, `p99_ms`, `min_ms`, `max_ms`, `sample_count`
- `throughput` -- `total_tokens`, `total_requests`, `tokens_per_sec`, `requests_per_sec`

```python
metrics = engine.get_metrics()
print(f"P95 latency: {metrics['latency']['p95_ms']:.1f}ms")
print(f"Throughput: {metrics['throughput']['tokens_per_sec']:.1f} tok/s")
```

---

### Other Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `check_server()` | `-> bool` | Check if llama-server is accessible at the configured URL |
| `check_for_updates()` | `@staticmethod` | Check GitHub for new versions (opt-in, silent on failure) |
| `reset_metrics()` | `-> None` | Reset all performance metric counters to zero |
| `unload_model()` | `-> None` | Unload the current model and stop the managed server |
| `get_last_suitability_report()` | `-> Optional[Dict]` | Return the last GGUF suitability report if `report_suitability=True` was used |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_loaded` | `bool` | Whether a model is currently loaded and ready |

---

## InferResult

Wrapper class for inference results with property-based access.

### Fields

| Property | Type | Description |
|----------|------|-------------|
| `success` | `bool` | Whether inference completed successfully |
| `text` | `str` | Generated text output |
| `tokens_generated` | `int` | Number of tokens generated |
| `latency_ms` | `float` | End-to-end inference latency in milliseconds |
| `tokens_per_sec` | `float` | Generation throughput (tokens per second) |
| `error_message` | `str` | Error description if `success` is `False` |

All fields are readable and writable via Python properties.

```python
result = engine.infer("Hello, world!")
print(str(result))          # Prints the generated text
print(repr(result))         # InferResult(tokens=42, latency=150.00ms, throughput=280.00 tok/s)
```

---

## Convenience Functions

### check_cuda_available

```python
def check_cuda_available() -> bool
```

Check if CUDA is available on the system. Delegates to `llamatelemetry.utils.detect_cuda()`.

**Returns:** `True` if CUDA is available, `False` otherwise.

---

### get_cuda_device_info

```python
def get_cuda_device_info() -> Optional[Dict[str, Any]]
```

Get CUDA device information.

**Returns:** Dictionary with `cuda_version` (str) and `gpus` (list of GPU info dicts), or `None` if CUDA is unavailable.

```python
info = llamatelemetry.get_cuda_device_info()
if info:
    print(f"CUDA {info['cuda_version']}, {len(info['gpus'])} GPU(s)")
```

---

### quick_infer

```python
def quick_infer(
    prompt: str,
    model_path: Optional[str] = None,
    max_tokens: int = 128,
    server_url: str = "http://127.0.0.1:8090",
    auto_start: bool = True,
) -> str
```

One-shot inference with minimal setup. Creates a temporary `InferenceEngine`, loads the model, runs inference, and returns the generated text.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | *required* | Input prompt |
| `model_path` | `Optional[str]` | `None` | Path to GGUF model (required if `auto_start=True`) |
| `max_tokens` | `int` | `128` | Maximum tokens to generate |
| `server_url` | `str` | `"http://127.0.0.1:8090"` | llama-server URL |
| `auto_start` | `bool` | `True` | Automatically start server if needed |

**Returns:** Generated text string, or an error message string on failure.

```python
text = llamatelemetry.quick_infer(
    "What is machine learning?",
    model_path="/path/to/model.gguf",
    max_tokens=200
)
print(text)
```

---

## Related Modules

- [Server and Models](server-models.md) -- ServerManager and model discovery
- [Client API](client-api.md) -- Low-level LlamaCppClient
- [Telemetry API](telemetry-api.md) -- OpenTelemetry integration
- [Multi-GPU and NCCL](multigpu-nccl.md) -- Multi-GPU configuration
