# Quantization & Unsloth API Reference

`llamatelemetry.quantization` provides quantization utilities for converting models to GGUF format
with NF4, Q4_K_M, Q5_K_M, and other schemes optimized for Tesla T4. `llamatelemetry.unsloth` provides
seamless integration between Unsloth fine-tuning and llamatelemetry inference, handling model loading,
LoRA adapter merging, and GGUF export.

```python
from llamatelemetry.quantization import (
    NF4Quantizer, NF4Config, quantize_nf4, dequantize_nf4,
    GGUFConverter, GGUFQuantType, convert_to_gguf, save_gguf, load_gguf_metadata,
    DynamicQuantizer, AutoQuantConfig, quantize_dynamic,
)
from llamatelemetry.unsloth import (
    UnslothModelLoader, load_unsloth_model, check_unsloth_available,
    UnslothExporter, ExportConfig, export_to_llamatelemetry, export_to_gguf,
    LoRAAdapter, AdapterConfig, merge_lora_adapters, extract_base_model,
)
```

---

## NF4Quantizer

4-bit NormalFloat quantizer compatible with bitsandbytes and QLoRA. NF4 is optimized for
normally distributed weights and provides better quality than uniform 4-bit quantization.

### NF4Config

```python
@dataclass
class NF4Config:
    blocksize: int = 64                  # Must be 64, 128, 256, or 512
    double_quant: bool = True            # Double quantization of absmax values
    quant_type: str = "nf4"
    compute_dtype: torch.dtype = torch.float16
```

### NF4Quantizer(blocksize, double_quant, compute_dtype)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `blocksize` | `int` | `64` | Block size for block-wise quantization |
| `double_quant` | `bool` | `True` | Apply secondary quantization to absmax values |
| `compute_dtype` | `torch.dtype` | `torch.float16` | Computation data type |

### NF4Quantizer.quantize()

```python
def quantize(self, weight: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `weight` | `torch.Tensor` | Input tensor (any shape) |

**Returns:** Tuple of:

- `quantized` -- `uint8` tensor with two 4-bit values packed per byte
- `state` -- Dict containing `absmax`, `code` (NF4 lookup table), `blocksize`, `shape`, `dtype`, `n_elements`, and optionally `state2` (double quantization state)

### NF4Quantizer.dequantize()

```python
def dequantize(self, quantized: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor
```

**Returns:** Dequantized tensor in original shape and dtype.

```python
quantizer = NF4Quantizer(blocksize=64, double_quant=True)
weight = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
qweight, state = quantizer.quantize(weight)
weight_restored = quantizer.dequantize(qweight, state)
print(f"Compression: {weight.nbytes / qweight.nbytes:.2f}x")
```

### quantize_nf4() / dequantize_nf4()

Convenience functions wrapping `NF4Quantizer`.

```python
def quantize_nf4(
    weight: torch.Tensor,
    blocksize: int = 64,
    double_quant: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]

def dequantize_nf4(
    quantized: torch.Tensor,
    state: Dict[str, Any],
) -> torch.Tensor
```

---

## GGUFConverter

Converts PyTorch models to GGUF (GPT-Generated Unified Format) for llama.cpp inference.

### GGUFQuantType

```python
class GGUFQuantType(Enum):
    F32 = 0       F16 = 1       BF16 = 30
    Q4_0 = 2      Q4_1 = 3      Q5_0 = 6      Q5_1 = 7
    Q8_0 = 8      Q8_1 = 9
    Q2_K = 10      Q3_K = 11     Q4_K = 12     Q5_K = 13     Q6_K = 14     Q8_K = 15
    Q4_K_S = 24    Q4_K_M = 25   Q5_K_S = 26   Q5_K_M = 27   Q6_K_S = 28   Q6_K_M = 29
    IQ2_XXS = 16   IQ2_XS = 17   IQ3_XXS = 18  IQ1_S = 19    IQ4_NL = 20
    IQ3_S = 21     IQ2_S = 22    IQ4_XS = 23
```

### GGUFConverter(model, tokenizer)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `Optional[Any]` | `None` | PyTorch model (transformers or Unsloth) |
| `tokenizer` | `Optional[Any]` | `None` | Associated tokenizer |

### GGUFConverter.convert()

```python
def convert(
    self,
    output_path: Union[str, Path],
    quant_type: Union[str, GGUFQuantType] = "Q4_K_M",
    verbose: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_path` | `Union[str, Path]` | -- | Output file path |
| `quant_type` | `Union[str, GGUFQuantType]` | `"Q4_K_M"` | Quantization type |
| `verbose` | `bool` | `True` | Print progress |

Extracts model metadata, converts tensor names to llama.cpp format, quantizes weights (embeddings and `lm_head` stay in F16), and writes GGUF v3 format.

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B")
converter = GGUFConverter(model)
converter.convert("model-q4_k_m.gguf", quant_type="Q4_K_M")
```

### GGUFConverter.add_metadata()

```python
def add_metadata(self, key: str, value: Any, value_type: Optional[GGUFValueType] = None)
```

Add a metadata key-value pair. Type is auto-inferred from the Python value if not provided.

### GGUFConverter.add_tensor()

```python
def add_tensor(self, name: str, tensor: torch.Tensor, quant_type: Union[str, GGUFQuantType] = "F16")
```

Add a tensor to be written. Converts to numpy and applies the specified quantization.

### GGUFConverter.extract_model_metadata()

```python
def extract_model_metadata(self)
```

Extracts `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `vocab_size`, `rope_theta`, and other config attributes, mapping them to GGUF metadata keys like `llama.embedding_length`, `llama.block_count`, etc.

### convert_to_gguf()

```python
def convert_to_gguf(
    model: Any,
    output_path: Union[str, Path],
    tokenizer: Optional[Any] = None,
    quant_type: str = "Q4_K_M",
    verbose: bool = True,
) -> Path
```

Convenience function. **Returns:** Path to saved GGUF file.

### save_gguf()

```python
def save_gguf(
    tensors: Dict[str, torch.Tensor],
    metadata: Dict[str, Any],
    output_path: Union[str, Path],
    quant_type: str = "F16",
)
```

Saves raw tensors and metadata as a GGUF file without requiring a full model object.

### load_gguf_metadata()

```python
def load_gguf_metadata(gguf_path: Union[str, Path]) -> Dict[str, Any]
```

Loads metadata from a GGUF file without loading tensor data. Uses the internal `GGUFReader`.

```python
metadata = load_gguf_metadata("model.gguf")
print(f"Architecture: {metadata.get('general.architecture')}")
print(f"Layers: {metadata.get('llama.block_count')}")
```

---

## DynamicQuantizer

Adaptive quantizer that selects optimal quantization based on model size, VRAM, and strategy.

### QuantStrategy

```python
class QuantStrategy(Enum):
    AGGRESSIVE = "aggressive"  # Q2_K, Q3_K -- maximum compression
    BALANCED   = "balanced"    # Q4_K_M -- recommended default
    QUALITY    = "quality"     # Q5_K_M, Q6_K -- higher quality
    MINIMAL    = "minimal"     # Q8_0, F16 -- minimal compression
```

### AutoQuantConfig

```python
@dataclass
class AutoQuantConfig:
    target_vram_gb: Optional[float] = None
    target_speed_tps: Optional[float] = None
    min_quality_ppl: Optional[float] = None
    strategy: QuantStrategy = QuantStrategy.BALANCED
    preserve_embeddings: bool = True
    preserve_output: bool = True
```

### DynamicQuantizer(target_vram_gb, strategy, device)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_vram_gb` | `Optional[float]` | `None` | Target VRAM usage (auto-detects via `nvidia-smi` if `None`) |
| `strategy` | `QuantStrategy` | `BALANCED` | Quantization strategy |
| `device` | `int` | `0` | CUDA device ID |

### DynamicQuantizer.recommend_config()

```python
def recommend_config(
    self,
    model: Optional[Any] = None,
    model_size_gb: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `Optional[Any]` | `None` | PyTorch model (optional if `model_size_gb` provided) |
| `model_size_gb` | `Optional[float]` | `None` | FP16 model size in GB |
| `verbose` | `bool` | `True` | Print recommendations |

**Returns:** Dict with `quant_type`, `expected_vram_gb`, `expected_speed_tps`, `compression_ratio`, `strategy`.

**Recommendation rules by strategy:**

| Strategy | 0-4 GB | 4-8 GB | 8-12 GB | 12+ GB |
|----------|--------|--------|---------|--------|
| Aggressive | Q4_K_S | Q3_K | Q2_K | Q2_K |
| Balanced | Q4_K_M | Q4_K_M | Q5_K_M | Q4_K_S |
| Quality | Q5_K_M | Q5_K_S | Q4_K_M | Q4_K_M |
| Minimal | Q8_0 | Q8_0 | Q6_K | Q6_K |

```python
quantizer = DynamicQuantizer(target_vram_gb=8.0)
config = quantizer.recommend_config(model_size_gb=3.5)
print(config['quant_type'])  # 'Q4_K_M'
```

### DynamicQuantizer.estimate_model_size_fp16()

```python
def estimate_model_size_fp16(self, model: Any) -> float
```

**Returns:** Estimated FP16 model size in GB.

### DynamicQuantizer.quantize_model()

```python
def quantize_model(
    self,
    model: Any,
    output_path: str,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
)
```

Quantizes model and writes GGUF file. Uses `recommend_config()` if `config` is `None`.

### quantize_dynamic()

```python
def quantize_dynamic(
    model: Any,
    output_path: str,
    target_vram_gb: Optional[float] = None,
    strategy: str = "balanced",
    verbose: bool = True,
) -> Dict[str, Any]
```

Convenience function. **Returns:** Configuration dict used.

---

## UnslothModelLoader

Loads Unsloth fine-tuned models with optional LoRA adapter handling.

### UnslothModelLoader(max_seq_length, load_in_4bit, dtype)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_seq_length` | `int` | `2048` | Maximum sequence length |
| `load_in_4bit` | `bool` | `True` | Load model in 4-bit quantization |
| `dtype` | `Optional[torch.dtype]` | `None` | Data type (auto-detects: bfloat16 for SM >= 8.0, float16 otherwise) |

### UnslothModelLoader.load()

```python
def load(
    self,
    model_name: str,
    adapter_path: Optional[str] = None,
    merge_adapters: bool = False,
) -> Tuple[Any, Any]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | -- | Model name or path (local or HuggingFace) |
| `adapter_path` | `Optional[str]` | `None` | Path to LoRA adapters |
| `merge_adapters` | `bool` | `False` | Merge adapters into base model |

**Returns:** Tuple of `(model, tokenizer)`.

### UnslothModelLoader.load_for_inference()

```python
def load_for_inference(
    self,
    model_name: str,
    adapter_path: Optional[str] = None,
) -> Tuple[Any, Any]
```

Loads model with adapters always merged and `FastLanguageModel.for_inference()` enabled.

### UnslothModelLoader.load_with_peft_config()

```python
def load_with_peft_config(
    self,
    model_name: str,
    peft_config: Dict[str, Any],
) -> Tuple[Any, Any]
```

Loads model and applies a PEFT/LoRA configuration dict using `get_peft_model()`.

### load_unsloth_model()

```python
def load_unsloth_model(
    model_name: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    adapter_path: Optional[str] = None,
    merge_adapters: bool = False,
) -> Tuple[Any, Any]
```

Convenience function wrapping `UnslothModelLoader`.

### check_unsloth_available()

```python
def check_unsloth_available() -> bool
```

**Returns:** `True` if the `unsloth` package is installed.

---

## UnslothExporter

Exports Unsloth models to GGUF format with quantization and optional LoRA merging.

### ExportConfig

```python
@dataclass
class ExportConfig:
    quant_type: str = "Q4_K_M"
    merge_lora: bool = True
    preserve_tokenizer: bool = True
    metadata: Optional[Dict[str, Any]] = None
    verbose: bool = True
    use_unsloth_native: bool = True
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `quant_type` | `str` | `"Q4_K_M"` | Quantization type for GGUF export |
| `merge_lora` | `bool` | `True` | Merge LoRA adapters before export |
| `preserve_tokenizer` | `bool` | `True` | Save tokenizer alongside GGUF |
| `metadata` | `Optional[Dict]` | `None` | Additional metadata to embed |
| `use_unsloth_native` | `bool` | `True` | Prefer Unsloth's native export method |

### UnslothExporter.export()

```python
def export(
    self,
    model: Any,
    tokenizer: Any,
    output_path: Union[str, Path],
    config: Optional[ExportConfig] = None,
) -> Path
```

**Returns:** Path to exported GGUF file. When `use_unsloth_native` is `True` and the model has `save_pretrained_gguf()`, uses Unsloth's native export. Otherwise falls back to llamatelemetry's `convert_to_gguf()`.

### UnslothExporter.export_with_unsloth_native()

```python
def export_with_unsloth_native(
    self,
    model: Any,
    tokenizer: Any,
    output_dir: Union[str, Path],
    quant_method: str = "q4_k_m",
)
```

### export_to_llamatelemetry() / export_to_gguf()

```python
def export_to_llamatelemetry(
    model, tokenizer, output_path,
    quant_type="Q4_K_M", merge_lora=True, verbose=True,
) -> Path

def export_to_gguf(
    model, tokenizer, output_path,
    quant_type="Q4_K_M", **kwargs,
) -> Path  # Alias for export_to_llamatelemetry
```

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("my_finetuned_model")
export_to_llamatelemetry(model, tokenizer, "model.gguf", quant_type="Q4_K_M")
```

---

## LoRAAdapter

Manager for LoRA adapters on Unsloth/PEFT models.

### AdapterConfig

```python
@dataclass
class AdapterConfig:
    adapter_name: str = "default"
    r: int = 16                                   # LoRA rank
    lora_alpha: int = 32
    target_modules: List[str] = None              # Defaults: q/k/v/o/gate/up/down_proj
    lora_dropout: float = 0.0
```

### LoRAAdapter(model)

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `Any` | Model with or without LoRA adapters |

Automatically detects adapter configuration from PEFT models on init.

### LoRAAdapter Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `has_adapters()` | `bool` | Check if adapters are present |
| `merge()` | `Any` | Merge LoRA adapters via `merge_and_unload()` |
| `extract_adapter_weights()` | `Dict[str, Tensor]` | Extract adapter weight tensors |
| `save_merged(merged_model, output_path, save_tokenizer, tokenizer)` | `None` | Save merged model to disk |
| `get_adapter_info()` | `Dict[str, Any]` | Get rank, alpha, target_modules, dropout |

```python
adapter = LoRAAdapter(model)
if adapter.has_adapters():
    info = adapter.get_adapter_info()
    print(f"LoRA rank: {info['rank']}, alpha: {info['alpha']}")
    merged = adapter.merge()
    adapter.save_merged(merged, "output_dir", tokenizer=tokenizer)
```

### merge_lora_adapters() / extract_base_model()

```python
def merge_lora_adapters(model: Any) -> Any
def extract_base_model(model: Any) -> Any
```

Convenience functions for merging adapters and unwrapping PEFT model wrappers.

---

## Related Documentation

- [GGUF API](gguf-api.md) -- GGUF file format utilities
- [CUDA & Inference API](cuda-inference-api.md) -- Tensor Core, FlashAttention
- [Core API](core-api.md) -- InferenceEngine for deployment
- [Quantization Guide](../guides/quantization.md)
- [Unsloth Guide](../guides/unsloth.md)
