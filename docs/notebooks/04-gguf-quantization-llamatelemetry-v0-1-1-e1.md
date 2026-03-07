# 04 GGUF Quantization and Suitability

Source: `notebooks/04-gguf-quantization-llamatelemetry-v0-1-1-e1.ipynb`


## Notebook focus

This page is a cell-by-cell walkthrough of the notebook, explaining the intent of each step and showing the exact code executed.


## Cell-by-cell walkthrough

### Cell 1 (Markdown)

# 04 GGUF Quantization and Suitability

Inspect GGUF model files, check GPU suitability, and explore the
quantization matrix.

**What you will learn:**
- Generate a GGUF metadata report
- Check model suitability for dual-T4 Kaggle setups
- View the full quantization type matrix
- Estimate GGUF file sizes for different quant types

**Requirements:** Kaggle notebook with a GGUF model dataset attached.

### Cell 2 (Markdown)

## 1) Install

### Cell 3 (Code)

**Summary:** Installs required dependencies and runtime tools.


```python
!pip -q install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1
```

### Cell 4 (Markdown)

## 2) GGUF report

`gguf_report()` reads the GGUF header and returns metadata including
architecture, parameter count, quantization type, and context length.

### Cell 5 (Code)

**Summary:** Imports core libraries: json, llamatelemetry. Works with GGUF models, quantization, or metadata.


```python
import json
from llamatelemetry.api.gguf import (
    gguf_report,
    report_model_suitability,
    quantization_matrix,
    estimate_gguf_size,
)

model_path = "/kaggle/input/your-model/model.gguf"

report = gguf_report(model_path)
print(json.dumps(report, indent=2, default=str))
```

### Cell 6 (Markdown)

## 3) Suitability check for dual T4

Checks whether the model fits in combined T4 VRAM (2 x 16 GB = 32 GB)
with the requested context size.

### Cell 7 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
suitability = report_model_suitability(model_path, ctx_size=4096, dual_t4=True)
print(json.dumps(suitability, indent=2, default=str))
```

### Cell 8 (Markdown)

## 4) Quantization matrix

View all 30+ GGUF quantization types with their bits-per-weight and
quality tier.

### Cell 9 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
matrix = quantization_matrix()
for row in matrix[:10]:  # first 10 types
    print(row)
```

### Cell 10 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
# As a pandas DataFrame (requires pandas)
try:
    df = quantization_matrix(as_dataframe=True)
    display(df)
except ImportError:
    print("pandas not available")
```

### Cell 11 (Markdown)

## 5) Size estimates

Estimate the file size for a given parameter count and quantization type.

### Cell 12 (Code)

**Summary:** Works with GGUF models, quantization, or metadata.


```python
param_counts = [1_000_000_000, 3_000_000_000, 7_000_000_000, 13_000_000_000]
quant_types = ["Q4_K_M", "Q5_K_M", "Q8_0"]

print(f"{'Params':>12s}  {'Quant':>8s}  {'Size (GB)':>10s}")
print("-" * 36)
for params in param_counts:
    for qt in quant_types:
        size_gb = estimate_gguf_size(params, qt) / (1024**3)
        print(f"{params/1e9:>10.1f}B  {qt:>8s}  {size_gb:>10.2f}")
```

### Cell 13 (Markdown)

## 6) Kaggle quantization recommendation

### Cell 14 (Code)

**Summary:** Imports core libraries: llamatelemetry. Works with GGUF models, quantization, or metadata.


```python
from llamatelemetry.api.gguf import recommend_quant_for_kaggle

rec = recommend_quant_for_kaggle(param_count=7_000_000_000, dual_t4=True)
print(f"Recommended quant: {rec}")
```
