# Multi-GPU and NCCL API Reference

The Multi-GPU and NCCL API provides everything needed to configure multi-GPU inference on Kaggle T4 x2, Colab, or any CUDA multi-device environment. It covers device enumeration, memory estimation, split-mode configuration, and NCCL collective communication.

**Modules:**

- `llamatelemetry.api.multigpu` — GPU detection, split-mode config, VRAM estimation
- `llamatelemetry.api.nccl` — NCCL version detection, environment setup, collective ops

---

## multigpu module

### SplitMode

Enum controlling how model weights are partitioned across multiple GPUs.

```python
from llamatelemetry.api.multigpu import SplitMode

class SplitMode(enum.Enum):
    NONE  = 0   # Single GPU — no splitting
    LAYER = 1   # Split by transformer layer (recommended for most models)
    ROW   = 2   # Split by tensor row (for very wide models)
```

| Value | Description |
|-------|-------------|
| `NONE` | All computation on a single GPU. Use when only one GPU is available. |
| `LAYER` | Consecutive transformer layers are assigned to different GPUs. Most compatible and efficient for dual-T4 setups. |
| `ROW` | Individual weight matrices are split by rows across GPUs via tensor parallelism. Requires NVLink for efficiency; not recommended on Kaggle PCIe T4. |

**Example:**

```python
from llamatelemetry.api.multigpu import SplitMode, MultiGPUConfig

config = MultiGPUConfig(
    n_gpu=2,
    split_mode=SplitMode.LAYER,
    tensor_split=[0.5, 0.5],
)
```

---

### GPUInfo

Dataclass holding properties for a single CUDA device.

```python
from llamatelemetry.api.multigpu import GPUInfo

@dataclass
class GPUInfo:
    index: int               # Device index (0-based)
    name: str                # Device name, e.g. "Tesla T4"
    total_memory: int        # Total VRAM in bytes
    free_memory: int         # Free VRAM in bytes at query time
    compute_capability: str  # e.g. "7.5" for Turing
    cuda_version: str        # CUDA runtime version string
```

**Example:**

```python
gpus = detect_gpus()
for gpu in gpus:
    print(f"GPU {gpu.index}: {gpu.name}")
    print(f"  VRAM: {gpu.total_memory / 1e9:.1f} GB total, "
          f"{gpu.free_memory / 1e9:.1f} GB free")
    print(f"  Compute: {gpu.compute_capability}")
```

---

### MultiGPUConfig

Dataclass that captures the full multi-GPU configuration passed to `load_model()` and `ServerManager`.

```python
from llamatelemetry.api.multigpu import MultiGPUConfig, SplitMode

@dataclass
class MultiGPUConfig:
    n_gpu: int = 1
    split_mode: SplitMode = SplitMode.LAYER
    main_gpu: int = 0
    tensor_split: Optional[List[float]] = None   # Per-GPU weight fractions, must sum to 1.0
    gpu_layers: Optional[int] = None             # Total layers to offload; None = auto
    n_parallel: int = 1                          # Parallel inference slots
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_gpu` | `int` | `1` | Number of GPUs to use |
| `split_mode` | `SplitMode` | `SplitMode.LAYER` | Layer or row split strategy |
| `main_gpu` | `int` | `0` | Primary GPU index used for embeddings and logits |
| `tensor_split` | `Optional[List[float]]` | `None` | Per-GPU weight fractions (e.g., `[0.5, 0.5]`). Defaults to equal split. |
| `gpu_layers` | `Optional[int]` | `None` | Total transformer layers to offload to GPU(s). `None` triggers auto-detection. |
| `n_parallel` | `int` | `1` | Number of parallel inference slots (for batching) |

**Example — custom Kaggle config:**

```python
config = MultiGPUConfig(
    n_gpu=2,
    split_mode=SplitMode.LAYER,
    main_gpu=0,
    tensor_split=[0.5, 0.5],
    n_parallel=2,
)
engine.load_model("llama-3.2-3b-instruct-Q4_K_M", multi_gpu_config=config)
```

---

## GPU Detection Functions

### detect_gpus

```python
def detect_gpus() -> List[GPUInfo]
```

Enumerate all available CUDA GPUs and return their properties. Returns an empty list if no CUDA devices are found or if CUDA is unavailable.

**Returns:** `List[GPUInfo]` — one entry per visible GPU.

```python
from llamatelemetry.api.multigpu import detect_gpus

gpus = detect_gpus()
print(f"Found {len(gpus)} GPU(s)")
for g in gpus:
    print(f"  [{g.index}] {g.name} — {g.total_memory // (1024**3)} GB")
```

---

### gpu_count

```python
def gpu_count() -> int
```

Return the number of available CUDA GPUs. Returns `0` on CPU-only environments.

```python
from llamatelemetry.api.multigpu import gpu_count

n = gpu_count()
print(f"GPUs available: {n}")
```

---

### get_cuda_version

```python
def get_cuda_version() -> Optional[str]
```

Return the CUDA runtime version string (e.g., `"12.5"`). Returns `None` if CUDA is not available.

```python
from llamatelemetry.api.multigpu import get_cuda_version

ver = get_cuda_version()
# "12.5" or None
```

---

## Memory Estimation Functions

### get_total_vram

```python
def get_total_vram(gpu_indices: Optional[List[int]] = None) -> int
```

Return total VRAM in bytes summed across all GPUs (or the specified subset).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpu_indices` | `Optional[List[int]]` | `None` | GPU indices to query; `None` queries all GPUs |

**Returns:** Total VRAM in bytes.

```python
from llamatelemetry.api.multigpu import get_total_vram

total = get_total_vram()
print(f"Total VRAM: {total / 1e9:.1f} GB")

# Query only GPU 0
vram_gpu0 = get_total_vram([0])
```

---

### get_free_vram

```python
def get_free_vram(gpu_indices: Optional[List[int]] = None) -> int
```

Return currently free VRAM in bytes summed across all GPUs (or the specified subset). Accounts for memory already allocated by the runtime.

```python
from llamatelemetry.api.multigpu import get_free_vram

free = get_free_vram()
print(f"Free VRAM: {free / 1e9:.1f} GB")
```

---

### estimate_model_vram

```python
def estimate_model_vram(
    model_size_b: float,
    quantization: str = "Q4_K_M",
    context_size: int = 4096,
    overhead_factor: float = 1.15,
) -> int
```

Estimate VRAM required to load and run a GGUF model.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_size_b` | `float` | *required* | Model parameter count in billions (e.g., `7.0` for a 7B model) |
| `quantization` | `str` | `"Q4_K_M"` | GGUF quantization type string (e.g., `"Q4_K_M"`, `"Q8_0"`, `"F16"`) |
| `context_size` | `int` | `4096` | KV cache context length in tokens |
| `overhead_factor` | `float` | `1.15` | Safety multiplier for KV cache and runtime overhead |

**Returns:** Estimated VRAM requirement in bytes.

```python
from llamatelemetry.api.multigpu import estimate_model_vram

# 13B model at Q4_K_M with 8K context
vram_needed = estimate_model_vram(13.0, "Q4_K_M", context_size=8192)
print(f"Estimated VRAM: {vram_needed / 1e9:.1f} GB")
```

---

### can_fit_model

```python
def can_fit_model(
    model_size_b: float,
    quantization: str = "Q4_K_M",
    context_size: int = 4096,
    gpu_indices: Optional[List[int]] = None,
) -> bool
```

Check whether the available VRAM is sufficient to run the specified model.

**Returns:** `True` if free VRAM ≥ estimated model VRAM.

```python
from llamatelemetry.api.multigpu import can_fit_model

if can_fit_model(7.0, "Q4_K_M", context_size=4096):
    print("Model fits in available VRAM")
else:
    print("Insufficient VRAM — try a smaller quantization")
```

---

### recommend_quantization

```python
def recommend_quantization(
    model_size_b: float,
    available_vram_gb: Optional[float] = None,
    context_size: int = 4096,
) -> str
```

Recommend the best GGUF quantization type for a given model size and available VRAM.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_size_b` | `float` | *required* | Model size in billions of parameters |
| `available_vram_gb` | `Optional[float]` | `None` | Available VRAM in GB. `None` queries live GPU memory. |
| `context_size` | `int` | `4096` | Target context window size |

**Returns:** Recommended quantization string, e.g. `"Q4_K_M"`, `"Q5_K_M"`, `"Q8_0"`.

```python
from llamatelemetry.api.multigpu import recommend_quantization

quant = recommend_quantization(13.0, available_vram_gb=15.0)
print(f"Recommended: {quant}")
# Example output: "Q4_K_M"
```

---

## Preset Configuration Functions

### kaggle_t4_dual_config

```python
def kaggle_t4_dual_config(
    model_size_b: float = 7.0,
    n_parallel: int = 2,
) -> MultiGPUConfig
```

Return a pre-tuned `MultiGPUConfig` for Kaggle's dual Tesla T4 (2 × 16 GB VRAM) environment.

Uses `SplitMode.LAYER` with a 50/50 tensor split and `main_gpu=0`.

```python
from llamatelemetry.api.multigpu import kaggle_t4_dual_config

config = kaggle_t4_dual_config(model_size_b=13.0, n_parallel=2)
engine.load_model("meta-llama-3.1-8b-instruct-Q4_K_M", multi_gpu_config=config)
```

---

### colab_t4_single_config

```python
def colab_t4_single_config(
    model_size_b: float = 7.0,
    n_parallel: int = 1,
) -> MultiGPUConfig
```

Return a pre-tuned `MultiGPUConfig` for Google Colab's single Tesla T4 (16 GB VRAM) environment. Uses `SplitMode.NONE` with `n_gpu=1`.

```python
from llamatelemetry.api.multigpu import colab_t4_single_config

config = colab_t4_single_config(model_size_b=7.0)
engine.load_model("gemma-3-4b-Q4_K_M", multi_gpu_config=config)
```

---

### auto_config

```python
def auto_config(
    model_size_b: Optional[float] = None,
    context_size: int = 4096,
    n_parallel: int = 1,
) -> MultiGPUConfig
```

Automatically detect available GPUs and return the optimal `MultiGPUConfig`. Detects Kaggle, Colab, and generic Linux environments.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_size_b` | `Optional[float]` | `None` | Model size for VRAM estimation; if `None`, uses a conservative default |
| `context_size` | `int` | `4096` | Target context size for KV cache estimation |
| `n_parallel` | `int` | `1` | Desired parallel slots |

**Returns:** Optimally configured `MultiGPUConfig`.

```python
from llamatelemetry.api.multigpu import auto_config

config = auto_config(model_size_b=13.0, context_size=8192, n_parallel=2)
print(f"Auto config: {config.n_gpu} GPU(s), mode={config.split_mode}")
```

---

## nccl module

NCCL (NVIDIA Collective Communications Library) support for GPU-to-GPU collective operations in multi-GPU setups. The `nccl` module wraps `libnccl.so.2` via ctypes and provides collective ops needed for model parallelism on dual-T4 Kaggle setups.

**Import:**

```python
from llamatelemetry.api.nccl import (
    NCCLCommunicator,
    is_nccl_available,
    get_nccl_version,
    get_nccl_info,
    print_nccl_info,
    setup_nccl_environment,
    kaggle_nccl_config,
)
```

---

### NCCLDataType

Enum mapping Python/NumPy types to NCCL's `ncclDataType_t`.

```python
class NCCLDataType(enum.Enum):
    INT8    = 0
    UINT8   = 1
    INT32   = 2
    UINT32  = 3
    INT64   = 4
    UINT64  = 5
    FLOAT16 = 6
    FLOAT32 = 7
    FLOAT64 = 8
    BFLOAT16 = 9
```

---

### NCCLResult

Enum for NCCL return codes.

```python
class NCCLResult(enum.Enum):
    SUCCESS          = 0
    UNHANDLED_CUDA   = 1
    SYSTEM_ERROR     = 2
    INTERNAL_ERROR   = 3
    INVALID_ARGUMENT = 4
    INVALID_USAGE    = 5
    REMOTE_ERROR     = 6
    IN_PROGRESS      = 7
```

---

### NCCLCommunicator

High-level context manager wrapping a NCCL communicator for collective operations across multiple GPUs.

```python
class NCCLCommunicator:
    def __init__(
        self,
        n_devs: int = 2,
        device_ids: Optional[List[int]] = None,
        lib_path: str = "libnccl.so.2",
    )
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_devs` | `int` | `2` | Number of GPUs to form the communicator over |
| `device_ids` | `Optional[List[int]]` | `None` | Explicit GPU indices (e.g., `[0, 1]`). `None` uses `range(n_devs)`. |
| `lib_path` | `str` | `"libnccl.so.2"` | Path to the NCCL shared library |

**Context manager:**

```python
with NCCLCommunicator(n_devs=2, device_ids=[0, 1]) as comm:
    # comm is initialized and ready
    comm.all_reduce(send_buf, recv_buf, count=1024, dtype=NCCLDataType.FLOAT32)
# Communicator is destroyed on exit
```

#### all_reduce

```python
def all_reduce(
    self,
    sendbuff: int,
    recvbuff: int,
    count: int,
    datatype: NCCLDataType = NCCLDataType.FLOAT32,
    op: int = 0,   # ncclSum
    stream: int = 0,
) -> NCCLResult
```

Perform an all-reduce collective: sum (or other reduction) across all GPUs, with result replicated to every GPU.

| Parameter | Description |
|-----------|-------------|
| `sendbuff` | GPU pointer (as int) to the input buffer |
| `recvbuff` | GPU pointer (as int) to the output buffer |
| `count` | Number of elements |
| `datatype` | Data type of elements |
| `op` | Reduction operation: `0` = sum, `1` = prod, `2` = max, `3` = min |
| `stream` | CUDA stream handle (0 = default stream) |

#### broadcast

```python
def broadcast(
    self,
    sendbuff: int,
    recvbuff: int,
    count: int,
    datatype: NCCLDataType = NCCLDataType.FLOAT32,
    root: int = 0,
    stream: int = 0,
) -> NCCLResult
```

Broadcast a buffer from `root` GPU to all other GPUs.

#### reduce

```python
def reduce(
    self,
    sendbuff: int,
    recvbuff: int,
    count: int,
    datatype: NCCLDataType = NCCLDataType.FLOAT32,
    op: int = 0,
    root: int = 0,
    stream: int = 0,
) -> NCCLResult
```

Reduce buffers from all GPUs to a single `root` GPU.

#### reduce_scatter

```python
def reduce_scatter(
    self,
    sendbuff: int,
    recvbuff: int,
    recvcount: int,
    datatype: NCCLDataType = NCCLDataType.FLOAT32,
    op: int = 0,
    stream: int = 0,
) -> NCCLResult
```

Reduce across all GPUs and scatter the result so each GPU receives a unique shard.

---

## NCCL Utility Functions

### is_nccl_available

```python
def is_nccl_available(lib_path: str = "libnccl.so.2") -> bool
```

Check whether NCCL is installed and loadable on this machine.

```python
from llamatelemetry.api.nccl import is_nccl_available

if is_nccl_available():
    print("NCCL is available")
else:
    print("NCCL not found — multi-GPU collective ops disabled")
```

---

### get_nccl_version

```python
def get_nccl_version(lib_path: str = "libnccl.so.2") -> Optional[str]
```

Return the NCCL version string (e.g., `"2.18.5"`). Returns `None` if NCCL is unavailable.

```python
from llamatelemetry.api.nccl import get_nccl_version

ver = get_nccl_version()
print(f"NCCL version: {ver}")  # "2.18.5"
```

---

### get_nccl_info

```python
def get_nccl_info() -> Dict[str, Any]
```

Return a dictionary with NCCL availability, version, and environment status.

```python
{
    "available": True,
    "version": "2.18.5",
    "lib_path": "libnccl.so.2",
    "nccl_p2p_disable": "0",
    "nccl_ib_disable": "1",
}
```

---

### print_nccl_info

```python
def print_nccl_info() -> None
```

Pretty-print NCCL diagnostic information to stdout. Useful for debugging multi-GPU setups.

```python
from llamatelemetry.api.nccl import print_nccl_info

print_nccl_info()
# NCCL Info
# ---------
# Available : True
# Version   : 2.18.5
# P2P       : enabled
# IB        : disabled
```

---

### setup_nccl_environment

```python
def setup_nccl_environment(
    disable_p2p: bool = False,
    disable_ib: bool = True,
    socket_ifname: str = "eth0",
    debug: Optional[str] = None,
) -> None
```

Set recommended NCCL environment variables for stable operation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disable_p2p` | `bool` | `False` | Set `NCCL_P2P_DISABLE=1` (use for PCIe-only setups without NVLink) |
| `disable_ib` | `bool` | `True` | Set `NCCL_IB_DISABLE=1` (recommended on Kaggle/Colab) |
| `socket_ifname` | `str` | `"eth0"` | Set `NCCL_SOCKET_IFNAME` for socket transport |
| `debug` | `Optional[str]` | `None` | Set `NCCL_DEBUG` level: `"INFO"`, `"WARN"`, `"TRACE"` |

```python
from llamatelemetry.api.nccl import setup_nccl_environment

# Kaggle T4 x2 (PCIe, no NVLink)
setup_nccl_environment(disable_p2p=True, disable_ib=True)
```

---

### kaggle_nccl_config

```python
def kaggle_nccl_config() -> Dict[str, str]
```

Return the recommended NCCL environment variable dictionary for Kaggle's dual-T4 PCIe setup.

```python
{
    "NCCL_P2P_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    "NCCL_SOCKET_IFNAME": "eth0",
    "NCCL_DEBUG": "WARN",
}
```

```python
from llamatelemetry.api.nccl import kaggle_nccl_config
import os

os.environ.update(kaggle_nccl_config())
```

---

## Complete Multi-GPU Example

```python
import llamatelemetry
from llamatelemetry.api.multigpu import (
    detect_gpus, auto_config, kaggle_t4_dual_config, recommend_quantization
)
from llamatelemetry.api.nccl import (
    is_nccl_available, setup_nccl_environment, print_nccl_info
)

# 1. Inspect available hardware
gpus = detect_gpus()
print(f"Detected {len(gpus)} GPU(s):")
for g in gpus:
    print(f"  [{g.index}] {g.name} — {g.total_memory // (1024**3)} GB VRAM")

# 2. Set up NCCL for dual-T4 PCIe
if is_nccl_available():
    setup_nccl_environment(disable_p2p=True, disable_ib=True)
    print_nccl_info()

# 3. Pick quantization based on available VRAM
quant = recommend_quantization(13.0)
print(f"Recommended quantization: {quant}")

# 4. Build multi-GPU config (auto-detect Kaggle vs local)
if len(gpus) >= 2:
    config = kaggle_t4_dual_config(model_size_b=13.0)
else:
    config = auto_config(model_size_b=7.0)

# 5. Run inference with multi-GPU config
with llamatelemetry.InferenceEngine() as engine:
    engine.load_model(
        f"meta-llama-3.1-8b-instruct-{quant}",
        multi_gpu_config=config,
    )
    result = engine.infer("Explain NCCL all-reduce in one sentence.")
    print(result.text)
```

---

## Related Documentation

- [CUDA and Inference API](cuda-inference-api.md) — CUDAGraph, TensorCore, FlashAttention
- [Kaggle API](kaggle-api.md) — KaggleEnvironment, split_gpu_session, presets
- [Server and Models](server-models.md) — ServerManager GPU configuration parameters
- [Guide: CUDA Optimizations](../guides/cuda-optimizations.md)
- [Guide: Kaggle Environment](../guides/kaggle-environment.md)
