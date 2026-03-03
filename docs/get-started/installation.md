# Installation

This guide covers every path to a working llamatelemetry installation: the
recommended one-line pip install, development installs from source, CUDA
prerequisites, GPU verification, optional dependency groups, environment
variables, container setups, and troubleshooting.

---

## Prerequisites

Before installing llamatelemetry, ensure your system meets the following
requirements.

### Python

llamatelemetry requires **Python >= 3.11**. Check your version:

```bash
python3 --version   # must be 3.11 or later
```

If your system Python is older, install 3.11+ via your package manager or
[pyenv](https://github.com/pyenv/pyenv):

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install python3.11 python3.11-venv python3.11-dev

# With pyenv
pyenv install 3.11.7
pyenv local 3.11.7
```

### CUDA toolkit and drivers

llamatelemetry targets **CUDA 12.x**. The NVIDIA driver must be version 525 or
later. Verify both:

```bash
nvidia-smi          # shows driver version and CUDA version
nvcc --version      # shows CUDA compiler version (if toolkit installed)
```

You need at least the NVIDIA driver and CUDA runtime libraries. The full CUDA
toolkit (with `nvcc`) is only required if you plan to build the C++/CUDA
extension from source.

### GPU compatibility

The SDK is production-tested on **Tesla T4** (SM 7.5, 16 GB VRAM). Any NVIDIA
GPU with compute capability >= 6.1 should work, but the model registry and
auto-configuration presets are tuned for T4-class hardware. Typical compatible
GPUs include:

| GPU | Compute Capability | VRAM |
|---|---|---|
| Tesla T4 | 7.5 | 16 GB |
| RTX 2080 Ti | 7.5 | 11 GB |
| RTX 3090 | 8.6 | 24 GB |
| RTX 4090 | 8.9 | 24 GB |
| A100 | 8.0 | 40/80 GB |
| L4 | 8.9 | 24 GB |

### Operating system

Linux is the primary supported platform (Ubuntu 20.04+ recommended). The SDK is
tested on Kaggle notebook images (Debian-based) and standard Ubuntu
installations. macOS and Windows are not officially supported but may work for
CPU-only experimentation.

---

## Install from GitHub (recommended)

The simplest installation pulls the tagged v0.1.0 release directly from GitHub:

```bash
pip install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

For a completely clean install that avoids cached wheels:

```bash
pip install --no-cache-dir --force-reinstall \
  git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

This installs the core package and all required dependencies (`numpy`,
`requests`, `huggingface_hub`, `tqdm`, `opentelemetry-api`, `opentelemetry-sdk`).

### Using a virtual environment

It is strongly recommended to use a virtual environment to avoid dependency
conflicts:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

---

## Install from source (development)

Clone the repository and install in editable mode for development:

```bash
git clone https://github.com/llamatelemetry/llamatelemetry.git
cd llamatelemetry
git checkout v0.1.0   # or main for latest

pip install -e ".[dev]"
```

The editable install (`-e`) allows you to modify source files without
reinstalling. The `[dev]` extra includes testing and formatting tools (pytest,
ruff, mypy).

### Building the C++/CUDA extension

The `llamatelemetry_cpp` pybind11 extension is built automatically by CMake
during installation if the CUDA toolkit is available. To build it manually:

```bash
cd csrc
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

The extension links against `cudart_static`, `cublas_static`, and
`cublasLt_static`. If CMake cannot find CUDA, set `CUDA_TOOLKIT_ROOT_DIR`:

```bash
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12
```

---

## Optional dependency groups

The `pyproject.toml` defines several extras for optional functionality. Install
them individually or combine them:

### Telemetry (OTLP exporters)

Required for exporting traces and metrics to Grafana Cloud, Jaeger, or any
OTLP-compatible backend:

```bash
pip install "llamatelemetry[telemetry] @ git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0"
```

This adds `opentelemetry-exporter-otlp-proto-http` and
`opentelemetry-exporter-otlp-proto-grpc`.

### Graphistry and RAPIDS

For graph visualization and GPU-accelerated graph analytics:

```bash
pip install "llamatelemetry[graphistry] @ git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0"
```

This adds `pygraphistry` and `pandas`.

### Jupyter

For notebook widgets and interactive visualization:

```bash
pip install "llamatelemetry[jupyter] @ git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0"
```

This adds `ipywidgets` and related display utilities.

### PyTorch and GPU monitoring

These are optional and installed separately since they have large footprints:

```bash
pip install torch pynvml    # for NCCL and GPU monitoring
pip install sseclient-py    # for SSE streaming support
pip install wandb           # for Weights & Biases logging
```

### All optional dependencies at once

```bash
pip install "llamatelemetry[telemetry,graphistry,jupyter,dev] @ git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0"
pip install torch pynvml sseclient-py wandb
```

---

## Verify installation

After installing, verify that the package loads and CUDA is visible:

```python
import llamatelemetry as lt

# Check version
print(f"llamatelemetry version: {lt.__version__}")  # expected: 0.1.0

# Check CUDA
cuda_info = lt.detect_cuda()
print(f"CUDA available: {cuda_info['available']}")
print(f"CUDA version:   {cuda_info['version']}")

for gpu in cuda_info["gpus"]:
    print(f"  GPU: {gpu['name']}")
    print(f"    Memory:             {gpu['memory']} MB")
    print(f"    Driver version:     {gpu['driver_version']}")
    print(f"    Compute capability: {gpu['compute_capability']}")
```

Expected output on a Tesla T4 system:

```
llamatelemetry version: 0.1.0
CUDA available: True
CUDA version:   12.2
  GPU: Tesla T4
    Memory:             15360 MB
    Driver version:     535.104.05
    Compute capability: 7.5
```

### Verify environment setup

The `setup_environment()` function configures paths for the llama-server binary
and CUDA libraries:

```python
from llamatelemetry import setup_environment

setup_environment()

import os
print(f"LLAMA_CPP_DIR:        {os.environ.get('LLAMA_CPP_DIR', 'not set')}")
print(f"LD_LIBRARY_PATH:      {os.environ.get('LD_LIBRARY_PATH', 'not set')}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
```

---

## Runtime bootstrap behavior

On first import, llamatelemetry automatically downloads runtime binaries and
shared libraries (approximately 961 MB). This is a one-time operation. The files
are stored inside the package directory:

| Directory | Contents | Approximate Size |
|---|---|---|
| `llamatelemetry/binaries/` | `llama-server` executable | ~200 MB |
| `llamatelemetry/lib/` | Shared libraries (CUDA, cuBLAS) | ~700 MB |
| `llamatelemetry/models/` | Downloaded GGUF model files | Varies per model |

The bootstrap runs automatically and shows a progress bar via `tqdm`. If the
download is interrupted, it resumes on the next import. To skip the bootstrap
(for example, if you have a pre-built llama.cpp), set the `LLAMA_SERVER_PATH`
environment variable to point to your binary.

---

## Environment variables

llamatelemetry reads the following environment variables. None are required for
basic usage.

| Variable | Purpose | Default |
|---|---|---|
| `LLAMA_SERVER_PATH` | Absolute path to a `llama-server` binary; skips bootstrap | Auto-discovered |
| `LLAMA_CPP_DIR` | Path to a llama.cpp build directory | Set by `setup_environment()` |
| `LD_LIBRARY_PATH` | Library search path; SDK prepends its own `lib/` directory | System default |
| `CUDA_VISIBLE_DEVICES` | Comma-separated GPU indices to expose | All GPUs visible |
| `HF_TOKEN` | Hugging Face token for gated model downloads | None |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint for telemetry export | None |
| `OTEL_EXPORTER_OTLP_HEADERS` | Authentication headers for OTLP export | None |
| `WANDB_API_KEY` | Weights & Biases API key for logging integration | None |

---

## Docker and container setup

For containerized deployments, use an NVIDIA CUDA base image and install
llamatelemetry on top:

```dockerfile
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create venv and install
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir \
    git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0

# Pre-run bootstrap to cache binaries in the image
RUN python -c "import llamatelemetry"

WORKDIR /workspace
CMD ["python3.11"]
```

Build and run with GPU access:

```bash
docker build -t llamatelemetry:v0.1.0 .
docker run --gpus all -it llamatelemetry:v0.1.0
```

The `--gpus all` flag requires the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
to be installed on the host.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'llamatelemetry'`

Ensure you are using the correct Python interpreter. If you installed in a
virtual environment, activate it first:

```bash
source .venv/bin/activate
python -c "import llamatelemetry; print(llamatelemetry.__version__)"
```

### `detect_cuda()` returns `available: False`

- Verify `nvidia-smi` runs successfully from the command line.
- Ensure `CUDA_VISIBLE_DEVICES` is not set to an empty string.
- Check that the NVIDIA driver is version 525 or later.
- In Docker, confirm the container was launched with `--gpus all`.

### Bootstrap download fails or stalls

- Check your network connection and firewall rules.
- If behind a corporate proxy, set `HTTP_PROXY` and `HTTPS_PROXY`.
- To retry, simply re-import the package. The download resumes from where it
  stopped.
- To skip bootstrap entirely, build llama.cpp from source and set
  `LLAMA_SERVER_PATH`.

### `ImportError` for optional dependencies

Optional modules gracefully degrade if their dependencies are not installed. If
you see an `ImportError` when using a specific feature, install the relevant
extras:

```bash
# For telemetry features
pip install opentelemetry-exporter-otlp-proto-http

# For graphistry features
pip install pygraphistry pandas

# For GPU monitoring
pip install pynvml

# For streaming
pip install sseclient-py
```

### CMake cannot find CUDA when building from source

Set the CUDA path explicitly:

```bash
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12
export PATH=$CUDA_TOOLKIT_ROOT_DIR/bin:$PATH
```

Then rebuild with `pip install -e .`

### Permission errors on `llamatelemetry/binaries/`

The bootstrap writes executables to the package directory. If installed
system-wide, the user may lack write permissions. Solutions:

- Install in a virtual environment (recommended).
- Set `LLAMA_SERVER_PATH` to a user-writable location.
- Run the initial import with appropriate permissions.

---

## Next steps

- [Quickstart](quickstart.md) -- load a model and run your first inference.
- [Kaggle Quickstart](kaggle-quickstart.md) -- optimized setup for Kaggle T4
  notebooks.
- [Server Management](../guides/server-management.md) -- advanced server
  configuration and lifecycle control.
