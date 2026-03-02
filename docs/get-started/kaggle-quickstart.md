# Kaggle Quickstart (Dual T4)

This workflow targets Kaggle notebooks with **GPU T4 x2**. `llamatelemetry` is optimized for this environment and can automatically configure split-GPU workloads.

## 1) Enable GPUs in Kaggle

- Notebook Settings → Accelerator → GPU (T4 x2)

## 2) Install

```python
!pip -q install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

## 3) Validate the environment

```python
from llamatelemetry import detect_cuda
print(detect_cuda())
```

## 4) Use Kaggle presets

```python
from llamatelemetry.api import kaggle_t4_dual_config

cfg = kaggle_t4_dual_config()
print(cfg)
```

## 5) Load a GGUF model

```python
import llamatelemetry as lt

engine = lt.InferenceEngine(enable_telemetry=False)
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)
```

## 6) Run inference

```python
result = engine.infer("Summarize the benefits of CUDA graphs.", max_tokens=96)
print(result.text)
```

## 7) Optional: telemetry and metrics

```python
engine = lt.InferenceEngine(
    enable_telemetry=True,
    telemetry_config={
        "service_name": "llamatelemetry-kaggle",
        "enable_llama_metrics": True,
    },
)
```

## 8) Recommended notebooks

- [01 Quickstart](../notebooks/01-quickstart-llamatelemetry-v0-1-0-e1.md)
- [03 Multi-GPU Inference](../notebooks/03-multi-gpu-inference-llamatelemetry-v0-1-0-e1.md)
- [06 Split-GPU Graphistry](../notebooks/06-split-gpu-graphistry-llamatelemetry-v0-1-0-e1.md)
- [09 Large Models Kaggle](../notebooks/09-large-models-kaggle-llamatelemetry-e3.md)
