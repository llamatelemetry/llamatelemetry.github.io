# GGUF API Reference

The GGUF modules provide comprehensive tools for inspecting, validating, quantizing, and analyzing GGUF (GPT-Generated Unified Format) model files. Two complementary modules are available: `llamatelemetry.api.gguf` for high-level utilities and CLI tool wrappers, and `llamatelemetry.gguf_parser` for zero-copy memory-mapped file reading.

**Modules:** `llamatelemetry.api.gguf`, `llamatelemetry.gguf_parser`

---

## GGUFReader (gguf_parser)

Memory-mapped GGUF file reader with zero-copy tensor access. Best for direct inspection of model files.

### Constructor

```python
class GGUFReader:
    def __init__(self, file_path: Union[str, Path])
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | `Union[str, Path]` | *required* | Path to GGUF file |

**Raises:** `FileNotFoundError` if the file does not exist. `ValueError` if the file has invalid GGUF magic or unsupported version.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `file_path` | `Path` | Path to the GGUF file |
| `metadata` | `Dict[str, Any]` | All metadata key-value pairs |
| `tensors` | `Dict[str, GGUFTensorInfo]` | Tensor name to info mapping |
| `alignment` | `int` | Data alignment (default: 32 bytes) |
| `tensor_data_offset` | `int` | Byte offset where tensor data begins |

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_tensor_data(tensor_name)` | `-> memoryview` | Get zero-copy memory-mapped view of tensor data |
| `get_metadata_value(key, default=None)` | `-> Any` | Get metadata value by key |
| `list_tensors()` | `-> List[str]` | Get all tensor names |
| `close()` | `-> None` | Close file and memory map |

### Context Manager

```python
from llamatelemetry.gguf_parser import GGUFReader

with GGUFReader("model.gguf") as reader:
    print(f"Model: {reader.metadata.get('general.name', 'unknown')}")
    print(f"Architecture: {reader.metadata.get('general.architecture')}")
    print(f"Tensors: {len(reader.tensors)}")

    for name, info in list(reader.tensors.items())[:5]:
        print(f"  {name}: {info.shape} ({info.ggml_type.name}, {info.nbytes / 1024**2:.1f} MB)")

    # Zero-copy tensor access
    data = reader.get_tensor_data("model.embed_tokens.weight")
```

### GGUFTensorInfo (gguf_parser)

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Tensor name |
| `n_dims` | `int` | Number of dimensions |
| `shape` | `List[int]` | Tensor shape |
| `ggml_type` | `GGMLType` | Quantization type |
| `offset` | `int` | Byte offset in data section |

| Property | Type | Description |
|----------|------|-------------|
| `numel` | `int` | Total number of elements |
| `nbytes` | `int` | Total size in bytes |

---

## GGMLType Enum

Tensor quantization types. Defined in both `llamatelemetry.api.gguf` and `llamatelemetry.gguf_parser`.

```python
from llamatelemetry.api.gguf import GGMLType
```

| Name | Value | Description |
|------|-------|-------------|
| `F32` | 0 | 32-bit float |
| `F16` | 1 | 16-bit float |
| `Q4_0` | 2 | 4-bit symmetric quantization |
| `Q4_1` | 3 | 4-bit asymmetric quantization |
| `Q5_0` | 6 | 5-bit symmetric |
| `Q5_1` | 7 | 5-bit asymmetric |
| `Q8_0` | 8 | 8-bit symmetric (near-lossless) |
| `Q2_K` | 10 | 2-bit K-quant |
| `Q3_K` | 11 | 3-bit K-quant |
| `Q4_K` | 12 | 4-bit K-quant |
| `Q5_K` | 13 | 5-bit K-quant |
| `Q6_K` | 14 | 6-bit K-quant |
| `IQ2_XXS` .. `IQ4_XS` | 16-23 | I-quant (vector quantization) |
| `BF16` | 29 | Brain float 16 |

---

## High-Level Report Functions

### gguf_report

```python
def gguf_report(
    model_path: str,
    include_raw_metadata: bool = False,
) -> Dict[str, Any]
```

Build a structured report for a GGUF model.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | *required* | Path to GGUF file |
| `include_raw_metadata` | `bool` | `False` | Include full raw metadata dict (can be large) |

**Returns:** Dictionary with keys: `path`, `size_gb`, `version`, `tensor_count`, `quantization_type`, `architecture`, `name`, `author`, `license`, `context_length`, `vocab_size`, `param_count_b`, `chat_template`, `summary`.

```python
from llamatelemetry.api.gguf import gguf_report
report = gguf_report("/models/gemma-3-1b-it-Q4_K_M.gguf")
print(f"Model: {report['name']}")
print(f"Size: {report['size_gb']} GB")
print(f"Architecture: {report['architecture']}")
print(f"Context: {report['context_length']}")
```

---

### report_model_suitability

```python
def report_model_suitability(
    model_path: str,
    ctx_size: int = 4096,
    dual_t4: bool = True,
) -> Dict[str, Any]
```

Assess whether a GGUF model fits Kaggle T4 GPUs and recommend settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | *required* | Path to GGUF file |
| `ctx_size` | `int` | `4096` | Target context length |
| `dual_t4` | `bool` | `True` | Use dual T4 VRAM (30GB) when True, single T4 (15GB) when False |

**Returns:** Dictionary with: `model`, `size_gb`, `quantization_type`, `param_count_b`, `context_length`, `ctx_size_target`, `available_vram_gb`, `estimated_kv_cache_gb`, `usable_vram_gb`, `fits`, `recommended_quant`, `recommended_gpu_layers`, `recommended_tensor_split`, `quantization_recommendation`.

```python
from llamatelemetry.api.gguf import report_model_suitability
report = report_model_suitability("/models/model.gguf", ctx_size=8192, dual_t4=True)
if report["fits"]:
    print(f"Model fits! Use {report['recommended_gpu_layers']} GPU layers")
else:
    print(f"Model too large. Try: {report['recommended_quant']}")
```

---

## Quantization Functions

### quantization_matrix

```python
def quantization_matrix(as_dataframe: bool = False) -> Union[List[Dict], "pd.DataFrame"]
```

Return a matrix of all quantization types with metadata.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `as_dataframe` | `bool` | `False` | Return as pandas DataFrame (requires pandas) |

**Returns:** List of dicts (or DataFrame) with keys: `name`, `generation`, `bits_per_weight`, `quality_score`, `requires_imatrix`, `description`.

```python
from llamatelemetry.api.gguf import quantization_matrix
for q in quantization_matrix():
    print(f"{q['name']:12s} {q['bits_per_weight']:.1f} bpw  quality={q['quality_score']}")
```

---

### estimate_gguf_size

```python
def estimate_gguf_size(param_count: int, quant_type: str) -> float
```

Estimate GGUF file size for a given model and quantization.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `param_count` | `int` | *required* | Number of model parameters |
| `quant_type` | `str` | *required* | Quantization type (e.g., `"Q4_K_M"`) |

**Returns:** Estimated file size in GB (includes ~10% overhead).

```python
from llamatelemetry.api.gguf import estimate_gguf_size
size = estimate_gguf_size(7_000_000_000, "Q4_K_M")
print(f"Estimated: {size:.1f} GB")  # ~4.0 GB
```

---

### quantize

```python
def quantize(
    input_path: str,
    output_path: str,
    quant_type: str = "Q4_K_M",
    n_threads: Optional[int] = None,
    allow_requantize: bool = False,
    leave_output_tensor: bool = False,
    pure: bool = False,
    imatrix_path: Optional[str] = None,
    include_weights: Optional[List[str]] = None,
    exclude_weights: Optional[List[str]] = None,
    output_tensor_type: Optional[str] = None,
    token_embedding_type: Optional[str] = None,
    llama_quantize_path: Optional[str] = None,
) -> bool
```

Quantize a GGUF model to a different precision. Wraps the `llama-quantize` CLI tool.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` | *required* | Input GGUF file |
| `output_path` | `str` | *required* | Output GGUF file |
| `quant_type` | `str` | `"Q4_K_M"` | Target quantization type |
| `imatrix_path` | `Optional[str]` | `None` | Importance matrix file for I-quants |
| `allow_requantize` | `bool` | `False` | Allow requantizing an already-quantized model |
| `pure` | `bool` | `False` | Disable k-quant mixtures |

**Returns:** `True` if successful.

---

### generate_imatrix

```python
def generate_imatrix(
    model_path: str,
    output_path: str,
    data_file: str,
    ctx_size: int = 512,
    n_batch: int = 512,
    n_threads: Optional[int] = None,
    n_gpu_layers: int = -1,
    llama_imatrix_path: Optional[str] = None,
) -> bool
```

Generate an importance matrix for higher-quality quantization. Wraps `llama-imatrix`.

```python
from llamatelemetry.api.gguf import generate_imatrix, quantize

# Generate importance matrix
generate_imatrix("model-f16.gguf", "model.imatrix", "calibration.txt")

# Use it for I-quant quantization
quantize("model-f16.gguf", "model-iq4xs.gguf", "IQ4_XS", imatrix_path="model.imatrix")
```

---

### convert_hf_to_gguf

```python
def convert_hf_to_gguf(
    model_path: str,
    output_path: Optional[str] = None,
    outtype: str = "f16",
    vocab_only: bool = False,
    pad_vocab: bool = False,
    skip_unknown: bool = False,
    metadata: Optional[Dict[str, str]] = None,
    python_path: str = "python3",
    convert_script: Optional[str] = None,
) -> bool
```

Convert a HuggingFace model to GGUF format. Wraps `convert_hf_to_gguf.py`.

---

### merge_lora

```python
def merge_lora(
    base_model: str,
    lora_path: str,
    output_path: str,
    lora_scale: float = 1.0,
    n_threads: Optional[int] = None,
    llama_export_path: Optional[str] = None,
) -> bool
```

Merge a LoRA adapter into a base GGUF model. Wraps `llama-export-lora`.

---

## Validation and Discovery

### validate_gguf

```python
def validate_gguf(path: str) -> Tuple[bool, str]
```

Validate a GGUF file by checking magic number, version, and tensor count.

**Returns:** Tuple of `(is_valid, message)`.

### get_model_summary

```python
def get_model_summary(path: str) -> str
```

Get a human-readable model summary string with architecture, size, parameters, context length, and metadata.

### compare_models

```python
def compare_models(path1: str, path2: str) -> Dict[str, Any]
```

Compare two GGUF models. Returns dict with `size_diff_gb`, `size_ratio`, `same_architecture`, `same_vocab`, `same_context`, and per-model info.

### find_gguf_models

```python
def find_gguf_models(directory: str, recursive: bool = True) -> List[str]
```

Find all GGUF files in a directory.

---

## Recommendation Functions

### recommend_quant_for_kaggle

```python
def recommend_quant_for_kaggle(
    param_count: int,
    dual_t4: bool = True,
    context_size: int = 4096,
    prefer_quality: bool = True,
) -> Dict[str, Any]
```

Recommend quantization for Kaggle T4 environments.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `param_count` | `int` | *required* | Number of model parameters (e.g., `7_000_000_000`) |
| `dual_t4` | `bool` | `True` | Use dual T4 (30GB) or single T4 (15GB) |
| `context_size` | `int` | `4096` | Target context window |
| `prefer_quality` | `bool` | `True` | Prefer quality over fitting larger models |

**Returns:** Dict with `quant_type`, `fits`, `estimated_size_gb`, `usable_vram_gb`, `headroom_gb`, `quality_score`, `requires_imatrix`, `description`, `generation`.

```python
from llamatelemetry.api.gguf import recommend_quant_for_kaggle
rec = recommend_quant_for_kaggle(70_000_000_000, dual_t4=True)
print(f"Recommended: {rec['quant_type']} ({rec['estimated_size_gb']} GB)")
print(f"Fits: {rec['fits']}, Quality: {rec['quality_score']}/10")
```

### get_recommended_quant

```python
def get_recommended_quant(original_size_gb: float, target_size_gb: float) -> str
```

Recommend quantization type to achieve a target file size. Returns the quantization name (e.g., `"Q4_K_M"`).

---

## Data Classes

### QuantTypeInfo

```python
@dataclass
class QuantTypeInfo:
    name: str              # e.g., "Q4_K_M"
    generation: str        # "legacy", "k-quant", or "i-quant"
    bits_per_weight: float # Approximate bits per weight
    quality_score: int     # 1-10 relative quality
    requires_imatrix: bool # Whether importance matrix recommended
    description: str       # Human-readable description
```

### GGUFMetadata

Structured metadata from GGUF file headers. Key fields: `general_architecture`, `general_name`, `context_length`, `embedding_length`, `block_count`, `head_count`, `vocab_size`, `chat_template`. Properties: `param_count`, `param_count_b`.

### GGUFModelInfo

Complete model information combining file stats, metadata, and tensor info. Key fields: `path`, `file_size`, `version`, `tensor_count`, `metadata`, `tensors`. Properties: `size_gb`, `quantization_type`.

---

## inspect_gguf (CLI)

```python
from llamatelemetry.gguf_parser import inspect_gguf
inspect_gguf("model.gguf")  # Prints detailed model info to stdout
```

Can also be run from the command line:

```bash
python -m llamatelemetry.gguf_parser model.gguf
```

---

## Related Modules

- [Server and Models](server-models.md) -- Model loading and management
- [Multi-GPU and NCCL](multigpu-nccl.md) -- GPU configuration for inference
