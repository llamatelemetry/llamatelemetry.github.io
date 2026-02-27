# Quantization Guide

`llamatelemetry` exposes quantization workflows through both `llamatelemetry.api.gguf` and `llamatelemetry.quantization`.

## GGUF utility path (`llamatelemetry.api.gguf`)

Common operations:

- Parse header: `parse_gguf_header`
- Quantize: `quantize`
- Convert HF to GGUF: `convert_hf_to_gguf`
- Merge LoRA: `merge_lora`
- Generate imatrix: `generate_imatrix`
- Validate and compare: `validate_gguf`, `compare_models`

Example:

```python
from llamatelemetry.api.gguf import parse_gguf_header, validate_gguf

info = parse_gguf_header("model.gguf", read_tensors=False)
ok, msg = validate_gguf("model.gguf")
```

## Quantization package path (`llamatelemetry.quantization`)

- NF4 helpers: `quantize_nf4`, `dequantize_nf4`
- GGUF conversion helpers: `convert_to_gguf`, `save_gguf`
- Dynamic quantization: `quantize_dynamic`

Example:

```python
from llamatelemetry.quantization import quantize_dynamic

q_model = quantize_dynamic(model)
```

## Model-size and fit planning

Use:

- `estimate_gguf_size`
- `get_recommended_quant`
- `recommend_quant_for_kaggle`

to choose a quantization strategy based on VRAM constraints and target quality.

## Best practice

1. Start with Q4_K_M for broad compatibility.
2. Validate output quality against baseline prompts.
3. Profile latency and memory before and after quantization changes.
