# 03 Multi-GPU Inference (Dual T4)

Source: `notebooks/03-multi-gpu-inference-llamatelemetry-v0-1-0-e1.ipynb`


## Notebook focus

This page is a cell-by-cell walkthrough of the notebook, explaining the intent of each step and showing the exact code executed.


## Cell-by-cell walkthrough

### Cell 1 (Markdown)

# 03 Multi-GPU Inference (Dual T4)

Configure layer-split and row-split inference across two GPUs using
`MultiGPUConfig`, `NCCLConfig`, and the `split_gpu_session` helper.

**What you will learn:**
- Configure `MultiGPUConfig` with `SplitMode.LAYER` and custom tensor splits
- Attach `NCCLConfig` for GPU communication
- Launch `InferenceEngine` in multi-GPU mode
- Use the `split_gpu_session` context manager

**Requirements:** Kaggle T4 x2 accelerator.

### Cell 2 (Markdown)

## 1) Install

### Cell 3 (Code)

**Summary:** Installs required dependencies and runtime tools.


```python
!pip -q install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

### Cell 4 (Markdown)

## 2) Detect GPUs

### Cell 5 (Code)

**Summary:** Imports core libraries: llamatelemetry.


```python
from llamatelemetry import detect_cuda

cuda_info = detect_cuda()
n_gpus = len(cuda_info.get('gpus', []))
print(f"GPUs detected: {n_gpus}")
for i, gpu in enumerate(cuda_info.get('gpus', [])):
    print(f"  GPU {i}: {gpu}")
```

### Cell 6 (Markdown)

## 3) Build MultiGPUConfig

| Field | Description |
|-------|-------------|
| `n_gpu_layers` | Layers to offload (-1 = all) |
| `split_mode` | `NONE`, `LAYER`, or `ROW` |
| `tensor_split` | VRAM fraction per GPU |
| `flash_attention` | Enable flash attention |
| `ctx_size` | Context window size |

### Cell 7 (Code)

**Summary:** Imports core libraries: llamatelemetry.


```python
from llamatelemetry.api.multigpu import MultiGPUConfig, SplitMode

multi_gpu = MultiGPUConfig(
    n_gpu_layers=-1,            # offload all layers
    split_mode=SplitMode.LAYER, # layer-wise split across GPUs
    tensor_split=[0.5, 0.5],    # equal VRAM allocation
    ctx_size=4096,
    batch_size=2048,
    ubatch_size=512,
    flash_attention=True,
)

print("CLI args:", multi_gpu.to_cli_args())
print("Dict:", multi_gpu.to_dict())
```

### Cell 8 (Markdown)

## 4) Build NCCLConfig

`NCCLConfig` sets environment variables for NCCL collective communication.

### Cell 9 (Code)

**Summary:** Imports core libraries: llamatelemetry.


```python
from llamatelemetry.api.nccl import NCCLConfig

nccl = NCCLConfig(gpu_ids=[0, 1])
print(f"GPU IDs: {nccl.gpu_ids}")
print(f"World size: {nccl.world_size}")
```

### Cell 10 (Markdown)

## 5) Launch InferenceEngine with multi-GPU

### Cell 11 (Code)

**Summary:** Imports core libraries: llamatelemetry. Creates or uses the high-level InferenceEngine to run GGUF inference. Works with GGUF models, quantization, or metadata. Loads a GGUF model (from registry, HF, or local path) and applies runtime settings. Runs inference and captures the generated output.


```python
import llamatelemetry as lt

model_path = "/kaggle/input/your-model/model.gguf"

engine = lt.InferenceEngine(enable_telemetry=False)
engine.load_model(
    model_path,
    auto_start=True,
    multi_gpu_config=multi_gpu,
    nccl_config=nccl,
    n_parallel=4,
)

result = engine.generate("Explain tensor split in llama.cpp", max_tokens=64)
print(f"Tokens/sec: {result.tokens_per_sec:.1f}")
print(result.text)
```

### Cell 12 (Markdown)

## 6) split_gpu_session helper

For workflows that need the LLM on one GPU and a graph/analytics workload on
the other, use the `split_gpu_session` context manager.

### Cell 13 (Code)

**Summary:** Imports core libraries: llamatelemetry.


```python
from llamatelemetry.kaggle import split_gpu_session

with split_gpu_session(llm_gpu=0, graph_gpu=1) as ctx:
    print("LLM server kwargs:", ctx["llm_server_kwargs"])
    # Use ctx["llm_server_kwargs"] when starting the server
    # Use ctx["graph_gpu"] for Graphistry / RAPIDS workloads
```

### Cell 14 (Markdown)

## 7) Cleanup

### Cell 15 (Code)

**Summary:** Cleans up or shuts down running resources.


```python
engine.unload_model()
print("Done.")
```
