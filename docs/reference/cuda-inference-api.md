# CUDA and Inference API

This page summarizes CUDA and inference optimization utilities.

## CUDA graphs

- `GraphCaptureConfig`
- `CUDAGraph`
- `GraphPool`
- `capture_graph()` / `replay_graph()` / `enable_cuda_graphs()`

## Tensor cores

- `TensorCoreConfig`
- `TensorCoreMatMul`
- `check_tensor_core_support()`
- `enable_tensor_cores()`
- `optimize_for_tensor_cores()`

## Triton kernels

- `KernelConfig`
- `TritonKernel`
- `register_kernel()` / `get_kernel()` / `list_kernels()`
- Example kernels: `triton_add`, `triton_layernorm`, `triton_softmax`

## Inference optimization

- `BatchConfig` / `BatchInferenceOptimizer`
- `ContinuousBatching`
- `KVCacheConfig` / `KVCache` / `PagedKVCache`
- `FlashAttentionConfig` and `enable_flash_attention()`

## Related docs

- [CUDA Optimizations Guide](../guides/cuda-optimizations.md)
