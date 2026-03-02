# Quantization

Quantization reduces model size and improves inference speed at the cost of some accuracy. `llamatelemetry` supports GGUF quantization workflows and recommendations.

## Key modules

- `llamatelemetry.quantization.dynamic` — dynamic quantization helpers
- `llamatelemetry.quantization.nf4` — NF4 quantization helpers
- `llamatelemetry.quantization.gguf` — GGUF quantization helpers
- `llamatelemetry.api.gguf` — GGUF analysis and quantization utilities

## Quantization recommendations

```python
from llamatelemetry.api.gguf import get_recommended_quant

print(get_recommended_quant(vram_gb=8, params_b=3))
```

## GGUF validation

```python
from llamatelemetry.api.gguf import validate_gguf

report = validate_gguf("/path/to/model.gguf")
print(report)
```

## Related docs

- [GGUF API](../reference/gguf-api.md)
- [Model Management](model-management.md)
