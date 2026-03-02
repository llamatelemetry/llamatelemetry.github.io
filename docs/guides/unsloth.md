# Unsloth Integration

`llamatelemetry.unsloth` provides a bridge between Unsloth fine-tuning workflows and GGUF export for inference.

## Key components

- `UnslothAdapter` — wraps Unsloth models and adapters
- `UnslothModelLoader` — checks availability and loads models
- `UnslothExporter` — exports to llamatelemetry or GGUF

## Example workflow

```python
from llamatelemetry.unsloth import UnslothModelLoader, UnslothExporter

loader = UnslothModelLoader()
model = loader.load_unsloth_model("unsloth/mistral-7b-bnb-4bit")

exporter = UnslothExporter()
exporter.export_to_gguf(model, output_path="mistral.gguf")
```

## Related docs

- [Quantization](quantization.md)
- [Unsloth API](../reference/quantization-unsloth.md)
