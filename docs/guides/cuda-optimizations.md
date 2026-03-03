# CUDA Optimizations

The `llamatelemetry.cuda` and `llamatelemetry.inference` modules provide GPU optimization utilities designed for maximum throughput on NVIDIA Tesla T4 (SM 7.5) and compatible hardware. This guide covers CUDA Graph capture and replay, Triton kernel integration, Tensor Core acceleration, FlashAttention for long contexts, KV cache management, and continuous batching strategies.

## Prerequisites

Before using these optimizations, ensure you have the required dependencies installed:

```bash
pip install torch triton flash-attn --no-build-isolation
```

Verify your GPU supports the features:

```python
import torch

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Compute capability: {major}.{minor}")
    print(f"Tensor Cores: {'Yes' if major >= 7 else 'No'}")
```

---

## CUDA Graph Capture and Replay

CUDA Graphs capture a sequence of GPU operations and replay them with minimal CPU overhead. On Tesla T4, this yields 20-40% latency reduction for small batch sizes by eliminating repeated kernel launch overhead.

### Basic Capture with Context Manager

```python
from llamatelemetry.cuda.graphs import CUDAGraph

graph = CUDAGraph()

# Capture operations
with graph.capture():
    output = model(input_tensor)

# Replay efficiently (no CPU overhead)
for _ in range(100):
    graph.replay()
```

### Capturing a Function

For explicit control, pass a callable directly to `capture()`. The graph runs warmup iterations before recording:

```python
from llamatelemetry.cuda.graphs import CUDAGraph, GraphCaptureConfig

config = GraphCaptureConfig(warmup_iters=5)
graph = CUDAGraph(config)

def forward_pass():
    return model(static_input)

output = graph.capture(forward_pass, warmup=True)

# Fast replay loop
for _ in range(1000):
    result = graph.replay()
```

### Managing Multiple Graphs with GraphPool

When you need different graphs for different input shapes or operations, `GraphPool` manages them by name:

```python
from llamatelemetry.cuda.graphs import GraphPool

pool = GraphPool()

# Capture graphs for different batch sizes
pool.capture("batch_1", lambda: model(input_batch_1))
pool.capture("batch_4", lambda: model(input_batch_4))

# Replay the right graph for the workload
result = pool.replay("batch_1")

# List and manage graphs
print(pool.list_graphs())  # ['batch_1', 'batch_4']
pool.remove("batch_1")
pool.clear()
```

### Convenience Function

For one-off graph captures, use the module-level helper:

```python
from llamatelemetry.cuda.graphs import capture_graph

graph = capture_graph(lambda: model(x), warmup_iters=3)
output = graph.replay()
```

!!! tip "When to Use CUDA Graphs"
    CUDA Graphs provide the greatest benefit for small batch sizes (1-8) where CPU-side kernel launch overhead dominates. For large batch sizes, the GPU is already saturated and the benefit is smaller.

!!! warning "Static Shapes Required"
    CUDA Graphs require fixed input and output tensor shapes. If your workload has dynamic shapes (variable sequence lengths), you need separate graphs per shape or should avoid graphs for those paths.

---

## Triton Kernel Integration

Triton lets you write GPU kernels in Python with performance comparable to hand-tuned CUDA. llamatelemetry ships built-in Triton kernels optimized for Tesla T4 and provides a registry for managing custom kernels.

### Built-in Kernels

Three optimized kernels are registered automatically when Triton is available:

```python
from llamatelemetry.cuda.triton_kernels import list_kernels

print(list_kernels())  # ['add', 'layernorm', 'softmax']
```

### Using High-Level API Functions

The simplest way to use Triton kernels is through the high-level wrapper functions. Each falls back to PyTorch automatically if Triton is not installed:

```python
from llamatelemetry.cuda.triton_kernels import triton_add, triton_layernorm, triton_softmax
import torch

# Element-wise addition
a = torch.randn(4096, device='cuda')
b = torch.randn(4096, device='cuda')
c = triton_add(a, b)

# Fused LayerNorm (combines mean, variance, normalize, affine in one kernel)
x = torch.randn(32, 768, device='cuda')
weight = torch.ones(768, device='cuda')
bias = torch.zeros(768, device='cuda')
normed = triton_layernorm(x, weight, bias, eps=1e-5)

# Numerically stable softmax
logits = torch.randn(32, 1024, device='cuda')
probs = triton_softmax(logits)
```

### Writing and Registering Custom Kernels

You can register your own Triton kernels with the global registry:

```python
import triton
import triton.language as tl
from llamatelemetry.cuda.triton_kernels import register_kernel, get_kernel, KernelConfig

@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.where(x > 0, x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

# Register with custom configuration
config = KernelConfig(name="relu", block_size=256, num_warps=4)
register_kernel("relu", relu_kernel, config)

# Use it later
kernel = get_kernel("relu")
kernel.launch(x, output, n_elements, grid=(n_blocks,))
```

!!! tip "Triton Kernel Tuning for T4"
    For Tesla T4, `block_size=128` and `num_warps=4` are good starting points. The fused LayerNorm kernel eliminates multiple memory round-trips, which is especially beneficial on T4's 320 GB/s memory bandwidth.

---

## Tensor Core Operations

Tesla T4's Tensor Cores provide up to 65 TFLOPS FP16 and 130 TOPS INT8 throughput -- a major speedup over standard CUDA cores for matrix-heavy workloads.

### Checking Tensor Core Support

```python
from llamatelemetry.cuda.tensor_core import check_tensor_core_support, get_tensor_core_info

# Simple check
if check_tensor_core_support():
    print("Tensor Cores available!")

# Detailed capabilities
info = get_tensor_core_info()
print(f"Architecture: {info.get('architecture')}")
print(f"FP16 TFLOPS: {info.get('fp16_tflops')}")
print(f"Estimated speedup: {info.get('estimated_speedup')}")
```

### Enabling Tensor Cores Globally

```python
from llamatelemetry.cuda.tensor_core import enable_tensor_cores

config = enable_tensor_cores(dtype=torch.float16, allow_tf32=True)
# All subsequent torch.matmul calls on FP16 tensors use Tensor Cores
```

### Tensor Core Matrix Multiplication

Use `matmul_tensor_core` for explicit FP16 Tensor Core matmul with automatic dtype conversion:

```python
from llamatelemetry.cuda.tensor_core import matmul_tensor_core

A = torch.randn(1024, 2048, device='cuda')  # FP32 input
B = torch.randn(2048, 4096, device='cuda')

# Converts to FP16, uses Tensor Cores, converts result back to FP32
C = matmul_tensor_core(A, B, dtype=torch.float16)
```

### Optimizing a Full Model

Apply Tensor Core optimizations to an entire PyTorch model for inference:

```python
from llamatelemetry.cuda.tensor_core import optimize_for_tensor_cores

model = MyModel()
model = optimize_for_tensor_cores(model, dtype=torch.float16)
# Model is now on CUDA, in FP16, with cuDNN benchmark mode enabled
output = model(input_tensor)
```

### Automatic Mixed Precision (AMP)

For training workflows, use AMP to automatically leverage Tensor Cores while maintaining numerical stability:

```python
from llamatelemetry.cuda.tensor_core import enable_amp

scaler, autocast = enable_amp(dtype=torch.float16)

for batch in dataloader:
    with autocast:
        output = model(batch)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## FlashAttention Integration

FlashAttention is an IO-aware attention algorithm that reduces memory bandwidth usage, enabling 2-3x speedups for sequences longer than 1024 tokens and significantly reducing memory consumption.

### Installation

```bash
pip install flash-attn --no-build-isolation
```

### Checking Availability

```python
from llamatelemetry.inference.flash_attn import check_flash_attention_available

if check_flash_attention_available():
    print("FlashAttention is ready")
```

### Enabling FlashAttention for a Model

```python
from llamatelemetry.inference.flash_attn import enable_flash_attention, FlashAttentionConfig

config = FlashAttentionConfig(
    version=2,
    causal=True,         # Autoregressive models
    dropout_p=0.0,       # No dropout for inference
    window_size=None,    # Full attention (or set tuple for sliding window)
)

model = enable_flash_attention(model, config)
```

### Direct Forward Pass

For custom attention implementations, call the FlashAttention forward pass directly. It falls back to standard attention if the library is not installed:

```python
from llamatelemetry.inference.flash_attn import flash_attention_forward

# Tensors must be [batch, seqlen, num_heads, head_dim] in FP16
q = torch.randn(2, 2048, 32, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 2048, 32, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 2048, 32, 64, device='cuda', dtype=torch.float16)

output = flash_attention_forward(q, k, v, causal=True)
```

### Estimating Optimal Context Length

Use the helper to determine how long your context can be given available VRAM:

```python
from llamatelemetry.inference.flash_attn import get_optimal_context_length

# Tesla T4 (16 GB VRAM), Gemma 3-1B model
ctx_len = get_optimal_context_length(
    model_size_b=1.0,
    available_vram_gb=12.0,
    use_flash_attention=True,
)
print(f"Recommended context length: {ctx_len}")  # ~8192
```

!!! tip "FlashAttention Memory Savings"
    Without FlashAttention, attention memory scales as O(n^2) with sequence length. FlashAttention reduces this to O(n), enabling 4K-8K contexts on T4 where standard attention would run out of memory.

---

## KV Cache Management

The KV cache stores key and value tensors from previous tokens during autoregressive generation, avoiding redundant computation.

### Basic KV Cache

```python
from llamatelemetry.inference.kv_cache import KVCache, KVCacheConfig

config = KVCacheConfig(
    max_batch_size=8,
    max_seq_length=4096,
    num_layers=32,
    num_heads=32,
    head_dim=128,
    dtype=torch.float16,
)

cache = KVCache(config)

# During generation, update cache per layer
for layer_idx in range(num_layers):
    k_cached, v_cached = cache.update(layer_idx, new_keys, new_values)
    # Use k_cached, v_cached for attention computation

# Retrieve cached values
cached = cache.get(layer_idx=0)
if cached:
    k, v = cached

# Clear between sequences
cache.clear()
```

### Paged KV Cache

For production workloads with many concurrent sequences, the paged KV cache reduces memory fragmentation (vLLM-style):

```python
from llamatelemetry.inference.kv_cache import PagedKVCache

paged_cache = PagedKVCache(config, page_size=16)
```

---

## Continuous Batching

Continuous batching allows new requests to join ongoing generation batches, maximizing GPU utilization. This is the same approach used by vLLM and llama-server.

### Batch Inference Optimizer

```python
from llamatelemetry.inference.batch import BatchInferenceOptimizer, BatchConfig

config = BatchConfig(
    max_batch_size=8,
    max_tokens=2048,
    dynamic_batching=True,
)

optimizer = BatchInferenceOptimizer(config)
results = optimizer.batch_infer(
    prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
    inference_fn=engine.infer,
)
```

### Convenience Function

```python
from llamatelemetry.inference.batch import batch_inference_optimized

results = batch_inference_optimized(
    prompts=["Hello", "World"],
    model=engine,
    max_batch_size=8,
)
```

!!! tip "Batch Size Tuning for T4"
    On Tesla T4 with 16 GB VRAM, start with `max_batch_size=4` for 7B models (Q4) and `max_batch_size=8` for 1B models. Monitor VRAM usage with `nvidia-smi` and increase the batch size until you approach 90% utilization.

---

## Performance Benchmarks

Typical performance improvements on Tesla T4 with a 1B parameter model (Q4_K_M quantization):

| Optimization | Latency Reduction | Memory Savings | Best For |
|---|---|---|---|
| CUDA Graphs | 20-40% | None | Small batches, repeated inference |
| Tensor Cores (FP16) | 2-4x throughput | 50% | Matrix-heavy operations |
| FlashAttention | 2-3x for long seqs | 5-10x | Sequences > 1024 tokens |
| Triton Fused LayerNorm | 1.5-2x | Minor | Transformer blocks |
| Continuous Batching | 2-4x throughput | None | Multi-user serving |

---

## Best Practices

1. **Combine optimizations** -- Enable Tensor Cores, FlashAttention, and CUDA Graphs together for maximum benefit.
2. **Profile first** -- Use `torch.profiler` or `nvidia-smi dmon` to identify bottlenecks before applying optimizations.
3. **Warm up the GPU** -- Always run warmup iterations before benchmarking to avoid cold-start overhead from JIT compilation.
4. **Match dtypes** -- Tensor Cores require FP16 inputs. Ensure your tensors are in `torch.float16` before matmul operations.
5. **Monitor VRAM** -- Use `torch.cuda.memory_summary()` to track memory usage and catch leaks early.

## Related Reference

- [CUDA and Inference API Reference](../reference/cuda-inference-api.md)
- [Quantization Guide](quantization.md)
- [Server Management](server-management.md)
