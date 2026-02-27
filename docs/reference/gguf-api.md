# GGUF API Reference

## Module: `llamatelemetry.api.gguf`

## Types and dataclasses

- `GGUFValueType`
- `GGMLType`
- `QuantTypeInfo`
- `GGUFMetadata`
- `GGUFTensorInfo`
- `GGUFModelInfo`

## Parsing and inspection

- `parse_gguf_header(path, read_tensors=False)`
- `find_gguf_models(directory, recursive=True)`
- `get_model_summary(path)`
- `compare_models(path1, path2)`
- `validate_gguf(path)`

## Quantization and conversion helpers

- `quantize(...)`
- `convert_hf_to_gguf(...)`
- `merge_lora(...)`
- `generate_imatrix(...)`
- `get_recommended_quant(...)`
- `estimate_gguf_size(...)`
- `recommend_quant_for_kaggle(...)`
- `print_quant_guide()`

## Example

```python
from llamatelemetry.api.gguf import (
    parse_gguf_header,
    validate_gguf,
    get_recommended_quant,
)

info = parse_gguf_header("model.gguf")
ok, msg = validate_gguf("model.gguf")
rec = get_recommended_quant(model_size_gb=4.0, vram_gb=15.0)
print(info.architecture, ok, rec)
```
