# CUDA & Inference API Reference

`llamatelemetry.cuda` provides low-level CUDA optimizations for Tesla T4 inference, including
CUDA Graphs, Triton custom kernels, and Tensor Core utilities. `llamatelemetry.inference` provides
advanced inference capabilities including FlashAttention integration, KV-cache optimization,
and batch inference strategies.

```python
from llamatelemetry.cuda import (
    CUDAGraph, GraphPool, capture_graph, replay_graph, enable_cuda_graphs,
    TritonKernel, register_kernel, get_kernel, list_kernels,
    TensorCoreConfig, enable_tensor_cores, matmul_tensor_core, check_tensor_core_support,
)
from llamatelemetry.inference import (
    FlashAttentionConfig, enable_flash_attention, flash_attention_forward,
    KVCache, KVCacheConfig, PagedKVCache, optimize_kv_cache,
    BatchInferenceOptimizer, ContinuousBatching, batch_inference_optimized,
)
```

---

## CUDAGraph

CUDA Graph wrapper for PyTorch operations. Captures a sequence of CUDA operations and replays them
with minimal CPU overhead, providing 20-40% latency reduction for small batch sizes on Tesla T4.

### CUDAGraph(config=None)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `Optional[GraphCaptureConfig]` | `None` | Capture configuration |

```python
@dataclass
class GraphCaptureConfig:
    pool: Optional[str] = None
    capture_error_mode: str = "thread_local"
    warmup_iters: int = 3
```

### CUDAGraph.capture()

```python
def capture(
    self,
    func: Optional[Callable] = None,
    inputs: Optional[Dict[str, torch.Tensor]] = None,
    warmup: bool = True,
) -> Any
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Optional[Callable]` | `None` | Function to capture (use context manager if `None`) |
| `inputs` | `Optional[Dict[str, Tensor]]` | `None` | Static input tensors |
| `warmup` | `bool` | `True` | Run warmup iterations before capture |

Can be used as a context manager or with an explicit function.

```python
# Context manager style
graph = CUDAGraph()
with graph.capture():
    output = model(input_tensor)

# Function style
graph = CUDAGraph()
graph.capture(lambda: model(input_tensor))
```

### CUDAGraph.replay()

```python
def replay(self) -> Any
```

Replay the captured graph. Raises `RuntimeError` if graph has not been captured.

**Returns:** Outputs from graph replay (tensor, dict, or tuple depending on capture).

```python
for _ in range(1000):
    output = graph.replay()  # Minimal CPU overhead
```

### CUDAGraph.is_captured() / CUDAGraph.reset()

```python
def is_captured(self) -> bool
def reset(self) -> None
```

---

## GraphPool

Pool of CUDA graphs for managing multiple graph instances with different operations or input shapes.

### GraphPool Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `capture(name, func, inputs, config)` | `str, Callable, ...` | `str` | Capture and store a named graph |
| `replay(name)` | `str` | `Any` | Replay graph by name |
| `get(name)` | `str` | `CUDAGraph` | Get graph by name |
| `remove(name)` | `str` | `None` | Remove and reset graph |
| `clear()` | -- | `None` | Clear all graphs |
| `list_graphs()` | -- | `List[str]` | List all graph names |

```python
pool = GraphPool()
pool.capture("forward", lambda: model(x))
pool.capture("backward", lambda: loss.backward())
output = pool.replay("forward")
```

---

## capture_graph() / replay_graph() / enable_cuda_graphs()

```python
def capture_graph(
    func: Callable,
    inputs: Optional[Dict[str, torch.Tensor]] = None,
    warmup_iters: int = 3,
) -> CUDAGraph

def replay_graph(graph: CUDAGraph) -> Any

def enable_cuda_graphs(model: torch.nn.Module) -> torch.nn.Module
```

`capture_graph` is a convenience function that creates a `CUDAGraph`, captures the function, and returns it.
`enable_cuda_graphs` wraps a model to use CUDA graphs when beneficial.

```python
graph = capture_graph(lambda: model(x), warmup_iters=3)
for _ in range(1000):
    output = graph.replay()
```

---

## TritonKernel

Wrapper for Triton JIT-compiled kernels with automatic grid computation and configuration.

### TritonKernel(name, kernel_func, config)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | -- | Kernel name |
| `kernel_func` | `Optional[Callable]` | `None` | Triton `@triton.jit` function |
| `config` | `Optional[KernelConfig]` | `None` | Kernel configuration |

```python
@dataclass
class KernelConfig:
    name: str
    block_size: int = 128
    num_warps: int = 4
    num_stages: int = 2
```

### TritonKernel.launch()

```python
def launch(self, *args, grid: Optional[Tuple[int, ...]] = None, **kwargs)
```

Launch the kernel. If `grid` is `None`, auto-computes from the first tensor argument's size.

### Kernel Registry Functions

```python
def register_kernel(name: str, kernel_func: Callable, config: Optional[KernelConfig] = None) -> TritonKernel
def get_kernel(name: str) -> Optional[TritonKernel]
def list_kernels() -> List[str]
```

### Built-in Kernels (Tesla T4 optimized)

When Triton is available, three kernels are auto-registered:

| Kernel | Name | Description |
|--------|------|-------------|
| `add_kernel` | `"add"` | Optimized element-wise addition |
| `fused_layernorm_kernel` | `"layernorm"` | Fused LayerNorm with RMS normalization |
| `softmax_kernel` | `"softmax"` | Numerically stable softmax |

### High-level Kernel Functions

```python
def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor
def triton_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor
def triton_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor
```

All functions fall back to PyTorch equivalents when Triton is not available.

```python
a = torch.randn(1024, device='cuda')
b = torch.randn(1024, device='cuda')
c = triton_add(a, b)  # Uses Triton kernel if available
```

---

## TensorCoreConfig

Configuration for Tensor Core operations on Tesla T4 (SM 7.5, Turing architecture).

```python
@dataclass
class TensorCoreConfig:
    enabled: bool = True
    dtype: torch.dtype = torch.float16
    allow_tf32: bool = True
    allow_fp16: bool = True
```

### check_tensor_core_support()

```python
def check_tensor_core_support(device: int = 0) -> bool
```

**Returns:** `True` if device supports Tensor Cores (SM >= 7.0). Tesla T4 is SM 7.5.

### enable_tensor_cores()

```python
def enable_tensor_cores(
    dtype: torch.dtype = torch.float16,
    allow_tf32: bool = True,
) -> TensorCoreConfig
```

Enables Tensor Core optimizations globally: sets `torch.backends.cuda.matmul.allow_tf32`, enables cuDNN benchmark mode.

**Returns:** `TensorCoreConfig` with applied settings.

### matmul_tensor_core()

```python
def matmul_tensor_core(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A` | `torch.Tensor` | -- | First matrix [M, K] |
| `B` | `torch.Tensor` | -- | Second matrix [K, N] |
| `out` | `Optional[torch.Tensor]` | `None` | Optional output tensor |
| `dtype` | `torch.dtype` | `torch.float16` | Computation dtype for Tensor Cores |

Converts inputs to FP16, performs matmul with Tensor Cores, converts back to original dtype.

```python
A = torch.randn(1024, 2048, device='cuda')
B = torch.randn(2048, 4096, device='cuda')
C = matmul_tensor_core(A, B)  # Fast FP16 Tensor Core matmul
```

### get_tensor_core_info()

```python
def get_tensor_core_info(device: int = 0) -> dict
```

**Returns:** Dict with `device`, `compute_capability`, `supported`, `architecture`, `fp16_tflops` (65 for T4), `int8_tops` (130 for T4), `estimated_speedup`.

### optimize_for_tensor_cores()

```python
def optimize_for_tensor_cores(model: torch.nn.Module, dtype: torch.dtype = torch.float16) -> torch.nn.Module
```

Moves model to CUDA, converts to FP16, enables eval mode and cuDNN benchmark.

### TensorCoreMatMul

Drop-in `torch.nn.Module` replacement for `torch.matmul` with automatic FP16 conversion.

```python
matmul = TensorCoreMatMul(dtype=torch.float16)
C = matmul(A, B)
```

---

## FlashAttentionConfig

Configuration for FlashAttention v2/v3 integration.

```python
@dataclass
class FlashAttentionConfig:
    enabled: bool = True
    version: int = 2
    causal: bool = True
    dropout_p: float = 0.0
    softmax_scale: Optional[float] = None
    window_size: Optional[Tuple[int, int]] = None  # Sliding window
```

### enable_flash_attention()

```python
def enable_flash_attention(
    model: torch.nn.Module,
    config: Optional[FlashAttentionConfig] = None,
) -> torch.nn.Module
```

Enables FlashAttention for a model. Returns the model unchanged if FlashAttention is not installed or already enabled.

### flash_attention_forward()

```python
def flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = True,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    window_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `torch.Tensor` | -- | `[batch, seqlen, num_heads, head_dim]` |
| `key` | `torch.Tensor` | -- | `[batch, seqlen, num_heads, head_dim]` |
| `value` | `torch.Tensor` | -- | `[batch, seqlen, num_heads, head_dim]` |
| `causal` | `bool` | `True` | Use causal masking |
| `dropout_p` | `float` | `0.0` | Dropout probability |
| `softmax_scale` | `Optional[float]` | `None` | Scale factor (default: 1/sqrt(d)) |
| `window_size` | `Optional[Tuple[int, int]]` | `None` | Sliding window size |

**Returns:** Attention output `[batch, seqlen, num_heads, head_dim]`. Falls back to standard attention when `flash-attn` is not installed.

```python
q = torch.randn(2, 2048, 32, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 2048, 32, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 2048, 32, 64, device='cuda', dtype=torch.float16)
output = flash_attention_forward(q, k, v, causal=True)
```

### get_optimal_context_length()

```python
def get_optimal_context_length(
    model_size_b: float,
    available_vram_gb: float,
    use_flash_attention: bool = True,
) -> int
```

Estimates optimal context length for given VRAM. With FlashAttention, scales linearly; without, scales quadratically.

```python
ctx = get_optimal_context_length(1.0, 12.0, use_flash_attention=True)
# Returns 8192 for 1B model with 12GB VRAM and FlashAttention
```

---

## KVCache / KVCacheConfig

Key-Value cache management for efficient sequential generation.

### KVCacheConfig

```python
@dataclass
class KVCacheConfig:
    max_batch_size: int = 8
    max_seq_length: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    dtype: torch.dtype = torch.float16
```

### KVCache(config)

```python
class KVCache:
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor,
               positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]
    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]
    def clear(self) -> None
```

The `update` method concatenates new key/value tensors with existing cached values for a layer. Returns the full cached key and value tensors.

### PagedKVCache

vLLM-style paged KV-cache for reduced memory fragmentation (simplified implementation).

```python
class PagedKVCache:
    def __init__(self, config: KVCacheConfig, page_size: int = 16): ...
```

### optimize_kv_cache()

```python
def optimize_kv_cache(model: torch.nn.Module) -> torch.nn.Module
```

Adds `_kv_cache` attribute to model for cache management.

---

## BatchInferenceOptimizer

Optimized batching strategies for maximizing throughput.

### BatchConfig

```python
@dataclass
class BatchConfig:
    max_batch_size: int = 8
    max_tokens: int = 2048
    dynamic_batching: bool = True
```

### BatchInferenceOptimizer(config)

```python
class BatchInferenceOptimizer:
    def batch_infer(
        self,
        prompts: List[str],
        inference_fn: Callable,
        **kwargs,
    ) -> List[Any]
```

Splits prompts into batches of `max_batch_size` and processes each batch sequentially.

```python
optimizer = BatchInferenceOptimizer(BatchConfig(max_batch_size=8))
results = optimizer.batch_infer(prompts, engine.infer)
```

---

## ContinuousBatching

Continuous batching for overlapping generation (vLLM-style, simplified).

```python
class ContinuousBatching:
    def __init__(self, max_batch_size: int = 8): ...
```

---

## batch_inference_optimized()

```python
def batch_inference_optimized(
    prompts: List[str],
    model: Any,
    max_batch_size: int = 8,
    **kwargs,
) -> List[Any]
```

Convenience function. Accepts any callable or object with an `infer` method.

```python
results = batch_inference_optimized(prompts, engine, max_batch_size=4)
```

---

## Related Documentation

- [Core API](core-api.md) -- InferenceEngine
- [Quantization & Unsloth](quantization-unsloth.md) -- Model quantization
- [Kaggle API](kaggle-api.md) -- GPU presets and context management
- [CUDA Optimizations Guide](../guides/cuda-optimizations.md)
