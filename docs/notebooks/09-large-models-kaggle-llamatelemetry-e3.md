# 09 Large Models on Kaggle

Source: `notebooks/09-large-models-kaggle-llamatelemetry-e3.ipynb`


## Notebook focus

This page is a cell-by-cell walkthrough of the notebook, explaining the intent of each step and showing the exact code executed.


## Cell-by-cell walkthrough

### Cell 1 (Markdown)

# 09 Large Models on Kaggle

Use server presets and suitability checks to run large GGUF models on
Kaggle's dual-T4 environment.

**What you will learn:**
- Check model suitability before loading
- Use `ServerPreset` for optimized server configuration
- Load large models with preset-derived kwargs

**Requirements:** Kaggle T4 x2 with a large GGUF model dataset.

### Cell 2 (Markdown)

## 1) Install

### Cell 3 (Code)

**Summary:** Installs required dependencies and runtime tools.


```python
!pip -q install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

### Cell 4 (Markdown)

## 2) Check GPU resources

### Cell 5 (Code)

**Summary:** Imports core libraries: llamatelemetry.


```python
from llamatelemetry import detect_cuda

cuda_info = detect_cuda()
print(f"GPUs: {len(cuda_info.get('gpus', []))}")
for gpu in cuda_info.get('gpus', []):
    print(f"  {gpu}")
```

### Cell 6 (Markdown)

## 3) Run suitability check

Before loading a large model, verify it fits in dual-T4 VRAM.

### Cell 7 (Code)

**Summary:** Imports core libraries: json, llamatelemetry. Works with GGUF models, quantization, or metadata.


```python
import json
from llamatelemetry.api.gguf import report_model_suitability

model_path = "/kaggle/input/your-large-model/model.gguf"

suitability = report_model_suitability(model_path, ctx_size=8192, dual_t4=True)
print(json.dumps(suitability, indent=2, default=str))
```

### Cell 8 (Markdown)

## 4) Available presets

| Preset | Target |
|--------|--------|
| `KAGGLE_DUAL_T4` | Kaggle dual T4 (32 GB total) |
| `KAGGLE_SINGLE_T4` | Kaggle single T4 (16 GB) |
| `COLAB_T4` | Colab T4 |
| `COLAB_A100` | Colab A100 |
| `LOCAL_3090` | Local RTX 3090 |
| `LOCAL_4090` | Local RTX 4090 |
| `CPU_ONLY` | CPU-only fallback |

### Cell 9 (Code)

**Summary:** Imports core libraries: llamatelemetry.


```python
from llamatelemetry.kaggle import ServerPreset, get_preset_config

preset = get_preset_config(ServerPreset.KAGGLE_DUAL_T4)
print(f"Preset: {ServerPreset.KAGGLE_DUAL_T4.value}")
print(f"Server kwargs: {preset.to_server_kwargs()}")
print(f"Load kwargs:   {preset.to_load_kwargs()}")
```

### Cell 10 (Markdown)

## 5) Load the model with preset config

### Cell 11 (Code)

**Summary:** Imports core libraries: llamatelemetry. Creates or uses the high-level InferenceEngine to run GGUF inference. Loads a GGUF model (from registry, HF, or local path) and applies runtime settings. Runs inference and captures the generated output.


```python
import llamatelemetry as lt

engine = lt.InferenceEngine(enable_telemetry=False)
engine.load_model(
    model_path,
    auto_start=True,
    **preset.to_load_kwargs(),
)

result = engine.generate("Summarize the model setup", max_tokens=64)
print(f"Tokens/sec: {result.tokens_per_sec:.1f}")
print(result.text)
```

### Cell 12 (Markdown)

## 6) Cleanup

### Cell 13 (Code)

**Summary:** Cleans up or shuts down running resources.


```python
engine.unload_model()
print("Done.")
```
