# Unsloth Integration Guide

`llamatelemetry.unsloth` supports moving from fine-tuning to deployable inference artifacts.

## Core exports

- Loader:
  - `check_unsloth_available`
  - `load_unsloth_model`
  - `UnslothModelLoader`
- Export:
  - `export_to_llamatelemetry`
  - `export_to_gguf`
  - `UnslothExporter`
- Adapters:
  - `LoRAAdapter`, `AdapterConfig`
  - `merge_lora_adapters`
  - `extract_base_model`

## Example workflow

```python
from llamatelemetry.unsloth import export_to_llamatelemetry

export_to_llamatelemetry(
    model=model,
    tokenizer=tokenizer,
    output_path="model.gguf",
    quant_type="Q4_K_M",
)
```

Then run inference with `InferenceEngine`:

```python
from llamatelemetry import InferenceEngine

engine = InferenceEngine()
engine.load_model("model.gguf", auto_start=True)
```

## Guidance

- Keep training and deployment artifact versions explicit.
- Validate generated GGUF with `llamatelemetry.api.gguf.validate_gguf`.
- Use representative benchmark prompts before promotion.
