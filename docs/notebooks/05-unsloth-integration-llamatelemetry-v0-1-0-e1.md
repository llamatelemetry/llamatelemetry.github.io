# 05 Unsloth Integration

Source: `notebooks/05-unsloth-integration-llamatelemetry-v0-1-0-e1.ipynb`


## Notebook focus

This page is a cell-by-cell walkthrough of the notebook, explaining the intent of each step and showing the exact code executed.


## Cell-by-cell walkthrough

### Cell 1 (Markdown)

# 05 Unsloth Integration

Fine-tune with Unsloth and export to GGUF for llama.cpp inference.

**What you will learn:**
- Load a model with `UnslothModelLoader`
- Configure GGUF export with `ExportConfig`
- Export a fine-tuned model to GGUF via `UnslothExporter`

**Requirements:** Kaggle notebook with GPU, `unsloth` and `torch` installed.
This notebook shows the API pattern. Actual fine-tuning requires a training
dataset and additional setup.

### Cell 2 (Markdown)

## 1) Install

### Cell 3 (Code)

**Summary:** Installs required dependencies and runtime tools.


```python
!pip -q install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
# Unsloth must be installed separately:
# !pip -q install unsloth
```

### Cell 4 (Markdown)

## 2) UnslothModelLoader

Wraps Unsloth model loading with 4-bit quantization and configurable
sequence length.

### Cell 5 (Code)

**Summary:** Imports core libraries: llamatelemetry.


```python
from llamatelemetry.unsloth import UnslothModelLoader, UnslothExporter, ExportConfig

# Create the loader (does not download anything yet)
loader = UnslothModelLoader(
    max_seq_length=2048,
    load_in_4bit=True,
)
print(f"Loader ready: seq_len={loader.max_seq_length}, 4bit={loader.load_in_4bit}")
```

### Cell 6 (Markdown)

## 3) Load a model for inference

**Note:** This cell requires `unsloth` and `torch` to be installed and
will download the model weights. Uncomment to run.

### Cell 7 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
# model_name = "unsloth/llama-3-8b-Instruct"
# model, tokenizer = loader.load_for_inference(model_name)
# print(f"Model type: {type(model).__name__}")
# print(f"Vocab size:  {len(tokenizer)}")
```

### Cell 8 (Markdown)

## 4) Configure GGUF export

| Field | Default | Description |
|-------|---------|-------------|
| `quant_type` | `Q4_K_M` | Target quantization type |
| `merge_lora` | `True` | Merge LoRA adapters before export |
| `preserve_tokenizer` | `True` | Include tokenizer in GGUF |
| `use_unsloth_native` | `True` | Use Unsloth's native export path |

### Cell 9 (Code)

**Summary:** Sets or updates environment variables for configuration.


```python
config = ExportConfig(
    quant_type="Q4_K_M",
    merge_lora=True,
    preserve_tokenizer=True,
    verbose=True,
)
print(f"Export config: quant={config.quant_type}, merge_lora={config.merge_lora}")
```

### Cell 10 (Markdown)

## 5) Export to GGUF

Uncomment after loading a model in step 3.

### Cell 11 (Code)

**Summary:** Works with GGUF models, quantization, or metadata.


```python
# exporter = UnslothExporter()
# output_path = exporter.export(
#     model,
#     tokenizer,
#     "model-q4.gguf",
#     config=config,
# )
# print(f"Exported to: {output_path}")
```

### Cell 12 (Markdown)

## 6) Verify the exported GGUF

### Cell 13 (Code)

**Summary:** Works with GGUF models, quantization, or metadata.


```python
# from llamatelemetry.api.gguf import gguf_report
# report = gguf_report(str(output_path))
# print(report)
```
