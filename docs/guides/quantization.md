# Quantization

Quantization reduces model size and improves inference speed by representing weights with lower-precision numbers. llamatelemetry provides tools for GGUF inspection, quantization analysis, size estimation, and dynamic quantization workflows.

## Overview

The quantization module includes:

- **gguf_report()** -- generate a comprehensive report of a GGUF model file
- **report_model_suitability()** -- check if a model fits on a specific GPU
- **quantization_matrix()** -- compare all quantization types for a given model size
- **estimate_gguf_size()** -- estimate file size for a given parameter count and quantization
- **GGUFReader** -- low-level GGUF file reader for metadata and tensor inspection
- **NF4Quantizer** -- NormalFloat4 quantization/dequantization
- **GGUFConverter** -- convert models to GGUF format
- **DynamicQuantizer** -- automatic quantization selection with QuantStrategy
- **recommend_quant_for_kaggle()** -- Kaggle-specific recommendations

## GGUF Quantization Types

GGUF supports over 30 quantization types. The most commonly used:

| Type | Bits/Weight | Quality | Size (7B) | Use Case |
|------|------------|---------|-----------|----------|
| `Q2_K` | ~2.6 | Low | ~2.8 GB | Maximum compression |
| `Q3_K_M` | ~3.4 | Fair | ~3.3 GB | Tight memory budgets |
| `Q4_0` | 4.0 | Good | ~3.8 GB | Fast inference, lower quality |
| `Q4_K_M` | ~4.8 | Very Good | ~4.4 GB | **Best general tradeoff** |
| `Q5_K_M` | ~5.5 | Excellent | ~5.0 GB | Higher quality |
| `Q6_K` | ~6.6 | Near FP16 | ~5.9 GB | Quality-sensitive tasks |
| `Q8_0` | 8.0 | Excellent | ~7.2 GB | Maximum quality |
| `F16` | 16.0 | Lossless | ~14 GB | No quantization |

!!! tip "Recommended Default"
    **Q4_K_M** provides the best balance of quality, speed, and memory usage for most workloads on Tesla T4 GPUs.

## GGUF Report

Generate a comprehensive report for any GGUF model file:

```python
from llamatelemetry.api.gguf import gguf_report

report = gguf_report("/path/to/model.gguf")

print(f"Architecture: {report['architecture']}")
print(f"Parameters: {report['parameters']}")
print(f"Quantization: {report['quantization']}")
print(f"Context length: {report['context_length']}")
print(f"Embedding size: {report['embedding_size']}")
print(f"Layers: {report['n_layers']}")
print(f"File size: {report['file_size_mb']:.1f} MB")
print(f"Vocab size: {report['vocab_size']}")
```

## Model Suitability

Check whether a model fits on your GPU:

```python
from llamatelemetry.api.gguf import report_model_suitability

suitability = report_model_suitability("/path/to/model.gguf", vram_gb=16)

print(f"Suitable: {suitability['suitable']}")
print(f"Estimated VRAM: {suitability['estimated_vram_gb']:.1f} GB")
print(f"GPU layers: {suitability['recommended_gpu_layers']}")
print(f"Context size: {suitability['recommended_ctx_size']}")
print(f"Headroom: {suitability['vram_headroom_gb']:.1f} GB")
```

The suitability check accounts for:

- Model weight memory
- KV cache memory at the recommended context size
- CUDA runtime overhead (~500 MB)
- A safety margin to prevent OOM errors

## Quantization Matrix

Compare all quantization types for a given model size:

```python
from llamatelemetry.api.gguf import quantization_matrix

matrix = quantization_matrix(parameters_b=7)

for quant_type, info in matrix.items():
    print(f"{quant_type:10s} | "
          f"Size: {info['estimated_size_gb']:5.1f} GB | "
          f"Quality: {info['relative_quality']:.2f} | "
          f"Fits T4: {info['fits_16gb']}")
```

Example output:

```
Q2_K       | Size:   2.8 GB | Quality: 0.65 | Fits T4: True
Q3_K_M     | Size:   3.3 GB | Quality: 0.75 | Fits T4: True
Q4_0       | Size:   3.8 GB | Quality: 0.82 | Fits T4: True
Q4_K_M     | Size:   4.4 GB | Quality: 0.88 | Fits T4: True
Q5_K_M     | Size:   5.0 GB | Quality: 0.93 | Fits T4: True
Q6_K       | Size:   5.9 GB | Quality: 0.97 | Fits T4: True
Q8_0       | Size:   7.2 GB | Quality: 0.99 | Fits T4: True
F16        | Size:  14.0 GB | Quality: 1.00 | Fits T4: True
```

## Size Estimation

Estimate GGUF file size without downloading the model:

```python
from llamatelemetry.api.gguf import estimate_gguf_size

# Estimate size for different model/quantization combinations
for params_b in [1, 3, 7, 13]:
    for quant in ["Q4_K_M", "Q8_0", "F16"]:
        size = estimate_gguf_size(parameters_b=params_b, quant_type=quant)
        print(f"{params_b}B {quant}: {size:.1f} GB")
```

## Kaggle Recommendations

Get quantization recommendations optimized for Kaggle T4 GPUs:

```python
from llamatelemetry.api.gguf import recommend_quant_for_kaggle

# Single T4 (16 GB)
rec = recommend_quant_for_kaggle(parameters_b=7, n_gpus=1)
print(f"Single T4: {rec}")

# Dual T4 (2x 16 GB)
rec = recommend_quant_for_kaggle(parameters_b=7, n_gpus=2)
print(f"Dual T4: {rec}")

# Larger model
rec = recommend_quant_for_kaggle(parameters_b=13, n_gpus=2)
print(f"13B on dual T4: {rec}")
```

## GGUFReader

Low-level reader for GGUF file metadata and tensor information:

```python
from llamatelemetry.api.gguf import GGUFReader

with GGUFReader("/path/to/model.gguf") as reader:
    # Access metadata
    print("Metadata:")
    for key, value in reader.metadata.items():
        print(f"  {key}: {value}")

    # List tensors
    print(f"\nTensors ({len(reader.tensors)}):")
    for tensor in reader.tensors[:10]:
        print(f"  {tensor.name}: shape={tensor.shape}, dtype={tensor.dtype}")

    # Read tensor data (for analysis)
    tensor_data = reader.get_tensor_data("token_embd.weight")
    print(f"\nEmbedding tensor shape: {tensor_data.shape}")
```

## NF4Quantizer

NormalFloat4 quantization for 4-bit quantized representations:

```python
from llamatelemetry.quantization.nf4 import NF4Quantizer
import numpy as np

quantizer = NF4Quantizer()

# Quantize a weight tensor
weights = np.random.randn(1024, 1024).astype(np.float32)
quantized = quantizer.quantize(weights)

print(f"Original size: {weights.nbytes / 1024:.0f} KB")
print(f"Quantized size: {quantized.nbytes / 1024:.0f} KB")
print(f"Compression ratio: {weights.nbytes / quantized.nbytes:.1f}x")

# Dequantize back to float32
reconstructed = quantizer.dequantize(quantized)
error = np.mean(np.abs(weights - reconstructed))
print(f"Mean absolute error: {error:.6f}")
```

## GGUFConverter

Convert models to GGUF format:

```python
from llamatelemetry.quantization.gguf import GGUFConverter

converter = GGUFConverter()

converter.convert(
    input_path="/path/to/model",
    output_path="/path/to/output.gguf",
    quant_type="Q4_K_M",
)
```

## DynamicQuantizer

Automatically select quantization parameters based on target constraints:

```python
from llamatelemetry.quantization.dynamic import DynamicQuantizer

quantizer = DynamicQuantizer()

# Get recommended configuration
config = quantizer.recommend_config(
    model_path="/path/to/model",
    target_vram_gb=16,
    target_quality="high",  # "low", "medium", "high", "maximum"
)

print(f"Recommended quant: {config['quant_type']}")
print(f"Estimated size: {config['estimated_size_gb']:.1f} GB")
print(f"Estimated quality: {config['estimated_quality']:.2f}")

# Apply quantization
quantizer.quantize_model(
    input_path="/path/to/model",
    output_path="/path/to/quantized.gguf",
    config=config,
)
```

## Best Practices

- **Use Q4_K_M as the default** -- it provides the best quality-per-bit for most models.
- **Check suitability first** with `report_model_suitability()` before loading on a GPU.
- **Use the quantization matrix** to compare options when choosing a quantization level.
- **On Kaggle T4**, use `recommend_quant_for_kaggle()` for automatically tuned settings.
- **Prefer pre-quantized models** from the registry or HuggingFace over quantizing yourself.
- **Test quality** by comparing outputs from different quantization levels on your specific task.

## Complete Example

```python
from llamatelemetry.api.gguf import (
    gguf_report,
    report_model_suitability,
    quantization_matrix,
    estimate_gguf_size,
    recommend_quant_for_kaggle,
    GGUFReader,
)

model_path = "/path/to/llama-3.1-8b-Q4_K_M.gguf"

# 1. Full report
report = gguf_report(model_path)
print(f"Model: {report['architecture']} ({report['parameters']})")
print(f"Quant: {report['quantization']}, Size: {report['file_size_mb']:.0f} MB")

# 2. Check T4 suitability
suit = report_model_suitability(model_path, vram_gb=16)
print(f"\nFits T4: {suit['suitable']} (est. {suit['estimated_vram_gb']:.1f} GB)")

# 3. Compare quantization options for 8B model
print("\nQuantization options for 8B:")
matrix = quantization_matrix(parameters_b=8)
for qt, info in list(matrix.items())[:5]:
    print(f"  {qt}: {info['estimated_size_gb']:.1f} GB")

# 4. Kaggle recommendation
rec = recommend_quant_for_kaggle(parameters_b=8, n_gpus=2)
print(f"\nKaggle dual-T4 recommendation: {rec}")

# 5. Inspect tensors
with GGUFReader(model_path) as reader:
    print(f"\nTensor count: {len(reader.tensors)}")
    print(f"Metadata keys: {len(reader.metadata)}")
```

## Related

- [Model Management](model-management.md) -- model registry and downloads
- [Unsloth Integration](unsloth.md) -- fine-tuning to GGUF export
- [GGUF API Reference](../reference/gguf-api.md)
