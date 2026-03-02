# GGUF API

`llamatelemetry.api.gguf` and `llamatelemetry.gguf_parser` provide utilities for inspecting, validating, and quantizing GGUF models.

## GGUF parser

```python
from llamatelemetry.gguf_parser import GGUFReader

with GGUFReader("model.gguf") as reader:
    print(reader.metadata)
    print(list(reader.tensors.keys())[:5])
```

Key classes:

- `GGUFReader` — memory-mapped GGUF reader
- `GGUFTensorInfo` — tensor metadata
- `GGMLType` / `GGUFValueType` — format enums

## GGUF utilities

Key functions in `llamatelemetry.api.gguf`:

- `parse_gguf_header()`
- `get_model_summary()` / `compare_models()`
- `validate_gguf()`
- `quantize()` / `merge_lora()` / `generate_imatrix()`
- `get_recommended_quant()` / `recommend_quant_for_kaggle()`
- `report_model_suitability()` / `gguf_report()`

## Related docs

- [Quantization Guide](../guides/quantization.md)
- [Model Management](../guides/model-management.md)
