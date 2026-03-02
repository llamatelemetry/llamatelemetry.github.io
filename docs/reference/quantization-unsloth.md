# Quantization and Unsloth API

## Quantization

`llamatelemetry.quantization` includes helper modules for dynamic quantization, NF4, and GGUF-specific conversions.

Key modules:

- `quantization.dynamic`
- `quantization.nf4`
- `quantization.gguf`

## Unsloth integration

`llamatelemetry.unsloth` provides adapters, loaders, and exporters.

Key classes and functions:

- `UnslothModelLoader` — check availability and load a model
- `UnslothAdapter` — wrap Unsloth fine-tuned models
- `UnslothExporter` — export to GGUF or llamatelemetry runtime
- `export_to_gguf(...)` — convert and export

## Related docs

- [Quantization Guide](../guides/quantization.md)
- [Unsloth Guide](../guides/unsloth.md)
