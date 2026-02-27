# CUDA and Inference API Reference

## Module: `llamatelemetry.cuda`

## CUDA graph APIs

- `GraphCaptureConfig`
- `CUDAGraph`
- `GraphPool`
- `capture_graph(...)`
- `replay_graph(...)`
- `enable_cuda_graphs(...)`

## Triton kernel APIs

- `KernelConfig`
- `TritonKernel`
- `register_kernel(...)`
- `get_kernel(name)`
- `list_kernels()`
- `triton_add(...)`
- `triton_layernorm(...)`
- `triton_softmax(...)`

## Tensor core APIs

- `TensorCoreConfig`
- `check_tensor_core_support(device=0)`
- `enable_tensor_cores(...)`
- `matmul_tensor_core(...)`
- `enable_amp(...)`
- `TensorCoreMatMul`
- `optimize_for_tensor_cores(...)`
- `get_tensor_core_info(device=0)`

---

## Module: `llamatelemetry.inference`

## Flash attention

- `FlashAttentionConfig`
- `enable_flash_attention(...)`
- `flash_attention_forward(...)`
- `check_flash_attention_available()`
- `get_optimal_context_length(...)`

## KV cache

- `KVCacheConfig`
- `KVCache`
- `PagedKVCache`
- `optimize_kv_cache(...)`

## Batch optimization

- `BatchConfig`
- `BatchInferenceOptimizer`
- `ContinuousBatching`
- `batch_inference_optimized(...)`
