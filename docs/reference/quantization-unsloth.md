# Quantization and Unsloth API Reference

## Module: `llamatelemetry.quantization`

### NF4

- `NF4Config`
- `NF4Quantizer`
- `quantize_nf4(...)`
- `dequantize_nf4(...)`

### GGUF conversion

- `GGUFQuantType`
- `GGUFConverter`
- `convert_to_gguf(...)`
- `save_gguf(...)`
- `load_gguf_metadata(...)`

### Dynamic quantization

- `QuantStrategy`
- `AutoQuantConfig`
- `DynamicQuantizer`
- `quantize_dynamic(...)`

---

## Module: `llamatelemetry.unsloth`

### Loader

- `check_unsloth_available()`
- `UnslothModelLoader`
- `load_unsloth_model(...)`

### Export

- `ExportConfig`
- `UnslothExporter`
- `export_to_llamatelemetry(...)`
- `export_to_gguf(...)`

### Adapters

- `AdapterConfig`
- `LoRAAdapter`
- `merge_lora_adapters(...)`
- `extract_base_model(...)`
