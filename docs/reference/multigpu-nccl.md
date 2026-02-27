# Multi-GPU and NCCL API Reference

## Module: `llamatelemetry.api.multigpu`

## Types

- `SplitMode`
- `GPUInfo`
- `MultiGPUConfig`

## Hardware detection and planning

- `detect_gpus()`
- `get_cuda_version()`
- `get_total_vram()`
- `get_free_vram()`
- `is_multi_gpu()`
- `gpu_count()`
- `estimate_model_vram(...)`
- `can_fit_model(...)`
- `recommend_quantization(...)`

## Preset helpers

- `kaggle_t4_dual_config(model_size_gb=7.0)`
- `colab_t4_single_config()`
- `auto_config()`

## Environment controls

- `set_cuda_visible_devices(*device_ids)`
- `get_cuda_visible_devices()`
- `print_gpu_info()`

---

## Module: `llamatelemetry.api.nccl`

## Types

- `NCCLResult`
- `NCCLDataType`
- `NCCLRedOp`
- `NCCLConfig`
- `NCCLInfo`
- `NCCLCommunicator`

## Functions

- `is_nccl_available()`
- `get_nccl_version()`
- `get_nccl_info()`
- `setup_nccl_environment(...)`
- `kaggle_nccl_config()`
- `print_nccl_info()`
- `get_llama_cpp_nccl_args(...)`

## Example

```python
from llamatelemetry.api import auto_config, get_nccl_info

cfg = auto_config()
print(cfg)
print(get_nccl_info())
```
