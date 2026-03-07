# Troubleshooting

This guide covers common issues you may encounter when installing, configuring, and using llamatelemetry. Each section describes symptoms, causes, and fixes.

---

## Installation Issues

### Package Fails to Install

**Symptoms:** `pip install llamatelemetry` fails with build errors.

**Fixes:**

- Ensure you are using Python 3.11 or later: `python3 --version`
- Upgrade pip and setuptools: `pip install --upgrade pip setuptools`
- If CMake build fails for the C++/CUDA extension, install CMake: `pip install cmake`
- On Kaggle, install in a single cell at the top of the notebook to avoid dependency conflicts

### C++/CUDA Extension Build Failure

**Symptoms:** Build errors mentioning `nvcc`, `cudart`, or `cublas` during installation.

**Fixes:**

- Verify CUDA toolkit is installed: `nvcc --version`
- Ensure `CUDA_HOME` or `CUDA_PATH` is set:
  ```bash
  export CUDA_HOME=/usr/local/cuda
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```
- For Tesla T4, confirm CUDA 12.x is installed (SM 7.5 support is required)
- If building from source, ensure CMake can find CUDA: `cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..`

### Import Errors After Install

**Symptoms:** `import llamatelemetry` fails with `ModuleNotFoundError` for dependencies.

**Fixes:**

- Install core dependencies: `pip install numpy requests huggingface_hub tqdm opentelemetry-api opentelemetry-sdk`
- For optional modules, install their specific dependencies:
  ```bash
  # Jupyter support
  pip install ipywidgets

  # Graphistry visualization
  pip install pygraphistry pandas

  # Unsloth fine-tuning
  pip install unsloth peft transformers

  # Triton kernels
  pip install triton

  # FlashAttention
  pip install flash-attn --no-build-isolation
  ```

---

## CUDA Detection Issues

### CUDA Not Detected

**Symptoms:**

- `detect_cuda()` returns `available: False`
- `torch.cuda.is_available()` returns `False`
- `llama-server` fails to start with GPU support

**Fixes:**

1. Verify the NVIDIA driver is installed and GPUs are visible:
   ```bash
   nvidia-smi
   ```
   If this command fails, install or update NVIDIA drivers.

2. Check that the CUDA toolkit version matches your driver:
   ```bash
   nvcc --version
   nvidia-smi  # Shows driver CUDA version in top-right
   ```

3. Verify you are not in a CPU-only container or virtual environment.

4. On Kaggle, ensure the notebook accelerator is set to **GPU (T4 x2)** in Settings.

5. If using PyTorch, ensure you have the CUDA-enabled build:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.version.cuda)
   ```

### Wrong GPU Selected

**Symptoms:** Operations run on GPU 0 when you expect GPU 1, or vice versa.

**Fixes:**

- Set the device explicitly:
  ```bash
  export CUDA_VISIBLE_DEVICES=0   # Use only GPU 0
  export CUDA_VISIBLE_DEVICES=1   # Use only GPU 1
  export CUDA_VISIBLE_DEVICES=0,1 # Use both
  ```
- In Python, specify the device:
  ```python
  import torch
  torch.cuda.set_device(1)
  ```

---

## Server Startup Problems

### `llama-server` Not Found

**Symptoms:**

- `ServerManager.find_llama_server()` returns `None`
- `InferenceEngine.load_model()` raises a runtime error about missing server binary

**Fixes:**

1. Set the path to your llama-server binary:
   ```bash
   export LLAMA_SERVER_PATH=/path/to/llama-server
   ```

2. If you built llama.cpp manually, set the build directory:
   ```bash
   export LLAMA_CPP_DIR=/path/to/llama.cpp/build/bin
   ```

3. Reinstall the package to trigger the bootstrap binary download:
   ```python
   import llamatelemetry
   # Bootstrap runs automatically on first import
   ```

4. Verify the binary is executable:
   ```bash
   chmod +x /path/to/llama-server
   /path/to/llama-server --help
   ```

### Server Fails to Start

**Symptoms:** Server process starts but immediately exits or hangs.

**Fixes:**

- Check the server log output for error messages
- Ensure the model file exists and is readable:
  ```python
  import os
  print(os.path.exists("model.gguf"))
  ```
- Verify sufficient VRAM is available:
  ```bash
  nvidia-smi
  ```
- Try starting with minimal options:
  ```python
  engine = InferenceEngine()
  engine.load_model("model.gguf", auto_start=True, n_gpu_layers=0)  # CPU only first
  ```
- Check if another process is using the default port (8080):
  ```bash
  lsof -i :8080
  ```
  Use a different port if needed:
  ```python
  engine.load_model("model.gguf", auto_start=True, port=8081)
  ```

### Server Responds with Errors

**Symptoms:** Server starts but returns HTTP 500 or empty responses.

**Fixes:**

- Wait for model loading to complete before sending requests (the engine handles this automatically, but manual server use requires patience)
- Check server health:
  ```python
  import requests
  response = requests.get("http://localhost:8080/health")
  print(response.json())
  ```
- Ensure the model format is correct (must be GGUF)
- Try reducing context size if the model is too large for available VRAM

---

## Out of Memory (OOM) Errors

### GPU OOM During Inference

**Symptoms:** `CUDA out of memory` errors during model loading or inference.

**Fixes:**

1. Use a more aggressively quantized model:
   ```python
   # Use Q4_K_M instead of Q8_0
   engine.load_model("model-Q4_K_M.gguf")
   ```

2. Reduce context length:
   ```python
   engine.load_model("model.gguf", context_size=2048)  # Instead of 4096
   ```

3. Reduce batch size:
   ```python
   result = engine.infer(prompt, max_tokens=128)  # Instead of 1024
   ```

4. Offload some layers to CPU:
   ```python
   engine.load_model("model.gguf", n_gpu_layers=20)  # Instead of all layers
   ```

5. Free unused GPU memory:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

6. Monitor VRAM usage:
   ```bash
   watch -n 1 nvidia-smi
   ```

### CPU OOM During Export

**Symptoms:** Process killed or `MemoryError` during GGUF export.

**Fixes:**

- LoRA adapter merging temporarily doubles memory usage. Use `load_in_4bit=True` to reduce the base footprint.
- Close other memory-intensive applications.
- On Kaggle, the notebook runtime has limited RAM. Consider exporting on a local machine with more memory.

### Tesla T4 VRAM Guidelines

| Model Size | Quantization | Approx. VRAM | Fits on T4 (16 GB)? |
|---|---|---|---|
| 1B | Q4_K_M | ~0.8 GB | Yes |
| 3B | Q4_K_M | ~2.2 GB | Yes |
| 7B | Q4_K_M | ~4.1 GB | Yes |
| 7B | Q8_0 | ~7.2 GB | Yes |
| 13B | Q4_K_M | ~7.4 GB | Yes (tight) |
| 13B | Q8_0 | ~14 GB | Barely |
| 70B | Q4_K_M | ~38 GB | No (need multi-GPU) |

---

## Multi-GPU Issues

### GPUs Not Detected

**Symptoms:** `MultiGPUConfig` shows fewer GPUs than expected.

**Fixes:**

- Verify all GPUs are visible:
  ```bash
  nvidia-smi
  ```
- Check `CUDA_VISIBLE_DEVICES` is not restricting visibility
- Ensure all GPUs have compatible drivers

### Split-Mode Errors

**Symptoms:** Multi-GPU inference fails or produces garbage output.

**Fixes:**

- Ensure both GPUs have the same architecture (e.g., both Tesla T4)
- Check that NCCL can communicate between GPUs:
  ```bash
  export NCCL_DEBUG=INFO
  ```
- Try layer split mode first (simpler than tensor split):
  ```python
  from llamatelemetry.api.multigpu import MultiGPUConfig, SplitMode

  config = MultiGPUConfig(split_mode=SplitMode.LAYER)
  ```
- On Kaggle dual-T4, use the recommended split-GPU session:
  ```python
  from llamatelemetry.kaggle import split_gpu_session
  ```

### NCCL Communication Failures

**Symptoms:** Hangs or errors mentioning NCCL during multi-GPU operations.

**Fixes:**

- Set NCCL environment variables for debugging:
  ```bash
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=ALL
  ```
- Ensure `LD_LIBRARY_PATH` includes the NCCL library directory
- Try disabling specific transports to isolate the issue:
  ```bash
  export NCCL_P2P_DISABLE=1  # Disable P2P (for debugging)
  ```

---

## Telemetry and OpenTelemetry Issues

### OpenTelemetry Not Available

**Symptoms:**

- `setup_telemetry()` returns `(None, None)`
- Telemetry spans are not being collected

**Fixes:**

Install the OpenTelemetry packages:

```bash
pip install opentelemetry-api opentelemetry-sdk
pip install opentelemetry-exporter-otlp-proto-grpc
pip install opentelemetry-exporter-otlp-proto-http
```

### OTLP Exporter Connection Refused

**Symptoms:** Telemetry data is not reaching your collector. Errors mentioning connection refused or timeout.

**Fixes:**

1. Verify the collector is running and accessible:
   ```bash
   curl -v http://localhost:4317  # gRPC endpoint
   curl -v http://localhost:4318  # HTTP endpoint
   ```

2. Check the endpoint configuration:
   ```python
   from llamatelemetry.telemetry import setup_telemetry

   tracer, meter = setup_telemetry(
       service_name="my-service",
       otlp_endpoint="http://localhost:4318",  # Verify this is correct
   )
   ```

3. If using Jaeger or Grafana, ensure their OTLP receivers are enabled.

### Missing Telemetry Attributes

**Symptoms:** Spans are created but `gen_ai.*` attributes are missing.

**Fixes:**

- Ensure you are using the instrumented inference methods (not raw HTTP calls)
- Check that the telemetry module was properly initialized before making inference calls
- Verify the OpenTelemetry SDK version is compatible (0.40+ recommended)

---

## Model Download Issues

### Registry Download Fails

**Symptoms:** Model download hangs, times out, or produces corrupted files.

**Fixes:**

1. Verify internet connectivity in your runtime environment
2. Provide a local GGUF path instead of a model name:
   ```python
   engine.load_model("/path/to/local/model.gguf")
   ```
3. For gated models, set your HuggingFace token:
   ```bash
   export HF_TOKEN=hf_your_token_here
   ```
   Or:
   ```python
   from huggingface_hub import login
   login(token="hf_your_token_here")
   ```
4. If the download was interrupted, delete the partial file and retry
5. Check available disk space -- GGUF files can be several gigabytes

### Bootstrap Binary Download Fails

**Symptoms:** First `import llamatelemetry` fails to download the llama-server binary (~961 MB).

**Fixes:**

- Ensure you have a stable internet connection and sufficient disk space
- Check if a proxy or firewall is blocking the download
- Set a custom download location:
  ```bash
  export LLAMATELEMETRY_HOME=/path/with/space
  ```
- Download the binary manually and set `LLAMA_SERVER_PATH`

---

## Missing Shared Libraries

**Symptoms:** `llama-server` or the C++ extension fails with errors about missing `.so` files (e.g., `libnccl.so`, `libcublas.so`).

**Fixes:**

1. Ensure `LD_LIBRARY_PATH` includes the llamatelemetry lib directory:
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/llamatelemetry/lib
   ```

2. Re-import llamatelemetry to re-run bootstrap:
   ```python
   import importlib
   import llamatelemetry
   importlib.reload(llamatelemetry)
   ```

3. Verify CUDA libraries are on the library path:
   ```bash
   ldconfig -p | grep cuda
   ldconfig -p | grep cublas
   ```

4. On Kaggle, the CUDA libraries are in `/usr/local/cuda/lib64` -- ensure this is in your path.

---

## Kaggle-Specific Issues

### Accelerator Not Set

**Symptoms:** No GPU detected in the Kaggle notebook.

**Fix:** Go to **Settings** (right sidebar) and set **Accelerator** to **GPU T4 x2**.

### Package Installation Order

**Symptoms:** Import errors or version conflicts after installing packages.

**Best Practice:** Install all packages in a single cell at the top of the notebook:

```python
!pip install -q llamatelemetry opentelemetry-api opentelemetry-sdk
```

Avoid running `pip install` in multiple cells, as this can cause dependency resolution issues.

### Disk Space Limits

**Symptoms:** Downloads fail or the kernel crashes due to insufficient disk space.

**Fixes:**

- Kaggle provides ~20 GB of disk space. Large models and binaries can exhaust this.
- Use smaller quantized models (Q4_K_M instead of Q8_0)
- Clean up temporary files:
  ```python
  !rm -rf /tmp/llamatelemetry_*
  ```
- Add model files as Kaggle datasets rather than downloading them each run

### Session Timeout

**Symptoms:** Long-running export or inference tasks are interrupted.

**Fix:** Kaggle sessions time out after extended idle periods. Keep the notebook active or break long tasks into smaller cells with intermediate saves.

---

## Common Error Messages

| Error | Cause | Fix |
|---|---|---|
| `RuntimeError: CUDA out of memory` | Model too large for VRAM | Use smaller model or more aggressive quantization |
| `ConnectionRefusedError: [Errno 111]` | Server not running | Start the server or check the port |
| `FileNotFoundError: llama-server` | Binary not found | Set `LLAMA_SERVER_PATH` |
| `ImportError: No module named 'unsloth'` | Unsloth not installed | `pip install unsloth` |
| `ImportError: No module named 'triton'` | Triton not installed | `pip install triton` |
| `json.JSONDecodeError` | Malformed server response | Check server health, restart if needed |
| `TimeoutError` | Server not responding | Increase timeout or check server load |
| `Warning: CUDA not available` | No GPU or wrong PyTorch build | Install CUDA-enabled PyTorch |

---

## Diagnostic Checklist

When reporting an issue, gather the following information:

```python
import sys
import torch

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

import llamatelemetry
print(f"llamatelemetry version: {llamatelemetry.__version__}")
```

```bash
nvidia-smi
nvcc --version
echo $CUDA_HOME
echo $LD_LIBRARY_PATH
echo $LLAMA_SERVER_PATH
```

---

## Getting Help

If the troubleshooting steps above do not resolve your issue:

- Check the [API Reference](../reference/index.md) for detailed parameter documentation
- Review the [Notebook Hub](../notebooks/index.md) for working examples
- Inspect the `tests/` directory in the source repository for runnable verification patterns
- Search existing issues on [GitHub](https://github.com/llamatelemetry/llamatelemetry/issues)
- File a new issue with the diagnostic checklist output above
