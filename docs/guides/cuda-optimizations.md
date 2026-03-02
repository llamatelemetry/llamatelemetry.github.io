# CUDA Optimizations

`llamatelemetry.cuda` and `llamatelemetry.inference` provide optional optimization utilities for CUDA inference workflows.

## CUDA graphs

```python
from llamatelemetry.cuda.graphs import enable_cuda_graphs

enable_cuda_graphs(model)
```

## Tensor cores

```python
from llamatelemetry.cuda.tensor_core import enable_tensor_cores

enable_tensor_cores(model)
```

## Triton kernels

```python
from llamatelemetry.cuda.triton_kernels import list_kernels

print(list_kernels())
```

## Inference optimizations

- Batch inference optimizers (`llamatelemetry.inference.batch`)
- KV-cache optimizations (`llamatelemetry.inference.kv_cache`)
- FlashAttention helpers (`llamatelemetry.inference.flash_attn`)

## Related reference

- [CUDA and Inference API](../reference/cuda-inference-api.md)
