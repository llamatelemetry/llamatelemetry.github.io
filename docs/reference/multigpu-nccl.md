# Multi-GPU and NCCL

## Multi-GPU configuration

`llamatelemetry.api.multigpu` provides helpers for device discovery, split modes, and recommended settings.

Key classes:

- `GPUInfo` — device properties
- `MultiGPUConfig` — config container
- `SplitMode` — enum for split strategies

Key functions:

- `detect_gpus()` / `gpu_count()`
- `get_total_vram()` / `get_free_vram()`
- `kaggle_t4_dual_config()` / `colab_t4_single_config()`
- `auto_config()` — automatic split and layer estimates
- `recommend_quantization()` / `estimate_model_vram()`

## NCCL integration

`llamatelemetry.api.nccl` provides lightweight NCCL discovery and environment helpers.

Key functions:

- `is_nccl_available()` / `get_nccl_version()`
- `get_nccl_info()` / `print_nccl_info()`
- `setup_nccl_environment()`
- `kaggle_nccl_config()`

## Related docs

- [CUDA and Inference API](cuda-inference-api.md)
- [Kaggle API](kaggle-api.md)
