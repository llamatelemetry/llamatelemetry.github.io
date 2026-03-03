# Unsloth Integration

The `llamatelemetry.unsloth` module provides a complete bridge between Unsloth fine-tuning workflows and llamatelemetry GGUF-based inference. It handles model loading, LoRA adapter management, and export to quantized GGUF files -- enabling a seamless pipeline from training to deployment.

## Workflow Overview

The typical fine-tuning to deployment pipeline:

1. **Fine-tune** a model with Unsloth (produces a model with LoRA adapters)
2. **Load** the fine-tuned model using `UnslothModelLoader`
3. **Manage adapters** with `LoRAAdapter` (inspect, merge, extract weights)
4. **Export** to GGUF using `UnslothExporter` with quantization
5. **Deploy** with `llamatelemetry.InferenceEngine` for fast inference

```python
from llamatelemetry.unsloth import (
    UnslothModelLoader, LoRAAdapter, UnslothExporter, ExportConfig,
    export_to_llamatelemetry,
)
```

---

## Prerequisites

Install Unsloth and its dependencies:

```bash
pip install unsloth peft transformers
```

Verify the installation:

```python
from llamatelemetry.unsloth import check_unsloth_available

if check_unsloth_available():
    print("Unsloth is ready")
else:
    print("Install Unsloth: pip install unsloth")
```

---

## Loading Unsloth Models

The `UnslothModelLoader` class handles loading models from local paths or HuggingFace Hub, with automatic dtype detection and optional LoRA adapter loading.

### Basic Loading

```python
from llamatelemetry.unsloth import UnslothModelLoader

loader = UnslothModelLoader(
    max_seq_length=2048,
    load_in_4bit=True,     # Recommended for T4 (16 GB VRAM)
    dtype=None,            # Auto-detect: bfloat16 for Ampere+, float16 for Turing
)

# Load from HuggingFace Hub
model, tokenizer = loader.load("unsloth/llama-3-8b-Instruct")

# Load from local path
model, tokenizer = loader.load("/path/to/local/model")
```

### Loading with LoRA Adapters

If you have saved LoRA adapters separately (e.g., from a fine-tuning run), load and optionally merge them:

```python
model, tokenizer = loader.load(
    "unsloth/llama-3-8b-Instruct",
    adapter_path="./my_lora_adapters",
    merge_adapters=True,   # Merge LoRA weights into base model
)
```

### Loading for Inference

The `load_for_inference` method automatically merges adapters and enables Unsloth's inference optimizations:

```python
model, tokenizer = loader.load_for_inference(
    "unsloth/llama-3-8b-Instruct",
    adapter_path="./my_adapters",
)
# Model is now merged and in inference mode
```

### Loading with Custom PEFT Config

Apply a new PEFT/LoRA configuration to a base model:

```python
model, tokenizer = loader.load_with_peft_config(
    "unsloth/llama-3-8b-Instruct",
    peft_config={
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.05,
    },
)
```

### Convenience Function

For quick loading without creating a loader instance:

```python
from llamatelemetry.unsloth import load_unsloth_model

model, tokenizer = load_unsloth_model(
    "unsloth/llama-3-8b-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    adapter_path="./adapters",
    merge_adapters=True,
)
```

!!! tip "4-bit Loading on Tesla T4"
    Always set `load_in_4bit=True` when working on Tesla T4. This reduces memory usage by roughly 4x, allowing you to load larger models within the 16 GB VRAM limit.

---

## LoRA Adapter Management

The `LoRAAdapter` class provides tools for inspecting, merging, and extracting LoRA adapter weights from fine-tuned models.

### Inspecting Adapters

```python
from llamatelemetry.unsloth import LoRAAdapter

adapter = LoRAAdapter(model)

# Check if model has adapters
if adapter.has_adapters():
    info = adapter.get_adapter_info()
    print(f"Adapter name: {info['adapter_name']}")
    print(f"LoRA rank: {info['rank']}")
    print(f"LoRA alpha: {info['alpha']}")
    print(f"Target modules: {info['target_modules']}")
    print(f"Dropout: {info['dropout']}")
```

### Merging Adapters

Merge LoRA weights into the base model. This is required before GGUF export:

```python
merged_model = adapter.merge()
```

### Extracting Adapter Weights

Inspect the raw LoRA weight tensors:

```python
weights = adapter.extract_adapter_weights()
print(f"Found {len(weights)} adapter tensors")

for name, tensor in list(weights.items())[:5]:
    print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
```

### Saving Merged Models

Save the merged model and tokenizer to disk in HuggingFace format:

```python
adapter.save_merged(
    merged_model,
    output_path="./merged_output",
    save_tokenizer=True,
    tokenizer=tokenizer,
)
```

### Convenience Functions

```python
from llamatelemetry.unsloth import merge_lora_adapters, extract_base_model

# Quick merge
merged = merge_lora_adapters(model)

# Extract base model from PEFT wrapper
base = extract_base_model(peft_model)
```

---

## Adapter Configuration

The `AdapterConfig` dataclass defines the LoRA configuration. When adapters are detected on a model, the config is populated automatically:

```python
from llamatelemetry.unsloth import AdapterConfig

# Default configuration (matches common Unsloth setups)
config = AdapterConfig(
    adapter_name="default",
    r=16,                  # LoRA rank
    lora_alpha=32,         # Scaling factor
    target_modules=[       # Modules with LoRA adapters
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.0,
)
```

!!! tip "Choosing LoRA Rank"
    For most fine-tuning tasks on T4, `r=16` provides a good balance between quality and memory. Increase to `r=32` or `r=64` for complex tasks. The effective learning rate scales as `lora_alpha / r`, so adjust `lora_alpha` proportionally.

---

## Exporting to GGUF

The `UnslothExporter` converts fine-tuned models to GGUF format with quantization, ready for llamatelemetry inference.

### Export Configuration

```python
from llamatelemetry.unsloth import ExportConfig

config = ExportConfig(
    quant_type="Q4_K_M",           # Quantization type
    merge_lora=True,               # Merge LoRA before export
    preserve_tokenizer=True,       # Save tokenizer alongside GGUF
    metadata={"author": "you"},    # Custom metadata
    verbose=True,                  # Print progress
    use_unsloth_native=True,       # Prefer Unsloth's built-in export
)
```

Available quantization types:

| Type | Size (7B model) | Quality | Use Case |
|---|---|---|---|
| `Q4_K_M` | ~4.1 GB | Good | Default, best balance |
| `Q5_K_M` | ~4.8 GB | Better | When VRAM allows |
| `Q8_0` | ~7.2 GB | Near-lossless | Quality-critical tasks |
| `Q4_K_S` | ~3.9 GB | Acceptable | Tight VRAM constraints |

### Using UnslothExporter

```python
from llamatelemetry.unsloth import UnslothExporter, ExportConfig

exporter = UnslothExporter()

config = ExportConfig(
    quant_type="Q4_K_M",
    merge_lora=True,
    preserve_tokenizer=True,
)

output_path = exporter.export(
    model=model,
    tokenizer=tokenizer,
    output_path="./output/model-q4.gguf",
    config=config,
)

print(f"Exported to: {output_path}")
```

The export process:

1. Checks if the model has LoRA adapters
2. Merges adapters if `merge_lora=True`
3. Extracts the base model from any wrappers
4. If `use_unsloth_native=True` and the model supports `save_pretrained_gguf`, uses Unsloth's built-in export
5. Otherwise, falls back to llamatelemetry's `quantization.convert_to_gguf`
6. Saves the tokenizer and metadata alongside the GGUF file

### Unsloth Native Export

When available, the Unsloth native export method is preferred as it handles model-specific details:

```python
exporter.export_with_unsloth_native(
    model=model,
    tokenizer=tokenizer,
    output_dir="./output",
    quant_method="q4_k_m",
)
```

If the native method fails, it automatically falls back to the llamatelemetry export pipeline.

### Convenience Functions

```python
from llamatelemetry.unsloth import export_to_llamatelemetry, export_to_gguf

# Full export with all options
path = export_to_llamatelemetry(
    model, tokenizer,
    output_path="model.gguf",
    quant_type="Q4_K_M",
    merge_lora=True,
    verbose=True,
)

# Alias (same function, for backward compatibility)
path = export_to_gguf(model, tokenizer, "model.gguf", quant_type="Q4_K_M")
```

---

## End-to-End Example

A complete workflow from fine-tuning to inference:

```python
# === Step 1: Fine-tune with Unsloth ===
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# ... (your training code here) ...

# === Step 2: Export to GGUF ===
from llamatelemetry.unsloth import export_to_llamatelemetry

export_to_llamatelemetry(
    model, tokenizer,
    output_path="my_finetuned_model.gguf",
    quant_type="Q4_K_M",
)

# === Step 3: Deploy with llamatelemetry ===
import llamatelemetry

engine = llamatelemetry.InferenceEngine()
engine.load_model("my_finetuned_model.gguf", auto_start=True)

result = engine.infer("Hello, how are you?", max_tokens=128)
print(result.text)
```

---

## Troubleshooting

### Unsloth Not Found

```
ImportError: Unsloth is not installed
```

Install Unsloth: `pip install unsloth`. On Kaggle, ensure you install it in the first cell before any other imports.

### PEFT/LoRA Import Errors

```
ImportError: No module named 'peft'
```

Install PEFT: `pip install peft`. This is required for adapter loading and merging.

### Export Fails with OOM

If GGUF export runs out of memory, the merge step temporarily doubles the model's memory footprint. Solutions:

- Use `load_in_4bit=True` to reduce base memory
- Export on a machine with more RAM (CPU RAM, not just GPU VRAM)
- Use a smaller quantization type like `Q4_K_S`

### Native Export Not Available

If `save_pretrained_gguf` is not available on your model, the exporter automatically falls back to llamatelemetry's built-in export pipeline. Ensure the `llamatelemetry.quantization` module is importable.

---

## Best Practices

1. **Always merge before export** -- Set `merge_lora=True` (the default) to ensure adapter weights are baked into the model.
2. **Use Q4_K_M as default** -- It provides the best balance of size and quality for T4 deployment.
3. **Preserve the tokenizer** -- Always set `preserve_tokenizer=True` so the GGUF file can be used with the correct tokenizer.
4. **Test before deploying** -- After export, load the GGUF with `InferenceEngine` and run a few test prompts to verify quality.
5. **Save adapter weights separately** -- Use `extract_adapter_weights()` to keep a backup of the LoRA weights before merging.

## Related Reference

- [Unsloth API Reference](../reference/quantization-unsloth.md)
- [Quantization Guide](quantization.md)
- [Inference Engine Guide](inference-engine.md)
- [Model Management Guide](model-management.md)
