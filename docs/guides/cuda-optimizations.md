# CUDA Optimizations Guide

`llamatelemetry.cuda` and `llamatelemetry.inference` expose advanced optimization APIs.

## CUDA module

- Graph APIs:
  - `CUDAGraph`, `GraphPool`, `capture_graph`, `replay_graph`, `enable_cuda_graphs`
- Triton APIs:
  - `TritonKernel`, `register_kernel`, `get_kernel`, `list_kernels`
- Tensor core APIs:
  - `TensorCoreConfig`, `enable_tensor_cores`, `matmul_tensor_core`

## Inference optimization module

- Flash attention:
  - `FlashAttentionConfig`, `enable_flash_attention`
- KV cache:
  - `KVCache`, `PagedKVCache`, `optimize_kv_cache`
- Batch optimization:
  - `BatchInferenceOptimizer`, `ContinuousBatching`, `batch_inference_optimized`

## Example: tensor core check

```python
from llamatelemetry.cuda import check_tensor_core_support

print(check_tensor_core_support(device=0))
```

## Example: flash attention enablement

```python
from llamatelemetry.inference import enable_flash_attention, FlashAttentionConfig

cfg = FlashAttentionConfig()
model = enable_flash_attention(model, config=cfg)
```

## Operational caution

- Many optimization paths are hardware- and dependency-dependent.
- Guard optional features with availability checks in production code.
- Fall back gracefully for unsupported GPU/runtime combinations.
