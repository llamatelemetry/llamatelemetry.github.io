---
title: Install llamatelemetry
description: Practical installation guide for llamatelemetry v0.1.1, including the recommended GitHub install path, optional extras, Kaggle notes, and post-install verification.
---

# Installation

This page focuses on the installation path that best matches the current
project state of `llamatelemetry` v0.1.1.

The SDK is a **Linux-first Python package** that bootstraps a bundled
`llama-server` workflow and is currently most aligned with **Kaggle dual-T4
notebooks** and nearby Linux environments. Some modules are broader than that,
but the package itself should be documented as **best supported on Kaggle and
Linux with NVIDIA GPUs**, not as a fully general-purpose cross-platform runtime.

## What to expect from the current package

Today, the package is strongest in these scenarios:

- Python 3.11+
- Linux
- NVIDIA GPU workflows
- local GGUF inference through `llama-server`
- Kaggle-focused helpers for dual Tesla T4 sessions
- optional OpenTelemetry-based observability

Treat Windows, macOS, and CPU-only use as experimental unless you validate your
exact workflow yourself.

## Prerequisites

### Python

Use **Python 3.11 or newer**.

```bash
python3 --version
```

A clean virtual environment is recommended:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### GPU and CUDA expectations

For the core CUDA-oriented workflow, you should have:

- an NVIDIA GPU
- working NVIDIA drivers
- a Linux environment where `nvidia-smi` works

```bash
nvidia-smi
```

You do **not** always need to compile CUDA code yourself. The package is built
around a bootstrap flow that tries to make the bundled runtime available for
you. Full CUDA toolchain setup is mainly relevant when you want to build pieces
from source or debug the lower-level C++/CUDA side.

## Recommended install

The most reliable documented path for the current project is installing directly
from the GitHub repo tag:

```bash
pip install --no-cache-dir --force-reinstall \
  git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1
```

That matches the package version exposed by the uploaded SDK snapshot.

## Optional extras

The package defines a few extras that are useful when you want richer notebook
or observability workflows.

### Telemetry extras

```bash
pip install "llamatelemetry[telemetry] @ git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1"
```

Use this when you want OTLP export or deeper OpenTelemetry workflows.

### Graphistry extras

```bash
pip install "llamatelemetry[graphistry] @ git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1"
```

Use this when you want graph visualization helpers.

### Jupyter extras

```bash
pip install "llamatelemetry[jupyter] @ git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1"
```

Use this for notebook-centric display helpers.

### Common add-ons installed separately

A few packages are referenced by the SDK but are best documented as separate
installs:

```bash
pip install torch pynvml sseclient-py wandb
```

Install these only when your workflow actually needs them.

## Kaggle install cell

For Kaggle, keep the first cell simple:

```python
!pip -q install --no-cache-dir --force-reinstall \
  git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1
```

Then add only the extra packages you need for that notebook.

## Development install

If you are editing the SDK itself:

```bash
git clone https://github.com/llamatelemetry/llamatelemetry.git
cd llamatelemetry
git checkout v0.1.1
pip install -e .
```

If you need the development toolchain too:

```bash
pip install -e ".[dev]"
```

## Post-install verification

Start with a minimal import and version check:

```python
import llamatelemetry as lt

print(lt.__version__)
```

Then verify the environment the package can see:

```python
import llamatelemetry as lt

cuda_info = lt.detect_cuda()
print(cuda_info)
```

And confirm the bundled `llama-server` path if bootstrap succeeded:

```python
import os

print(os.environ.get("LLAMA_SERVER_PATH"))
```

## First smoke test

A practical first smoke test is to create an engine and inspect it before you
load any model:

```python
import llamatelemetry as lt

engine = lt.InferenceEngine(server_url="http://127.0.0.1:8080")
print(engine.server_url)
```

Once that works, continue with the [Quickstart](quickstart.md) or the
[Kaggle Quickstart](kaggle-quickstart.md).

## Known documentation boundaries

To keep this page honest, these points are important:

- the SDK snapshot clearly targets **Kaggle dual-T4 workflows** as its most
  opinionated runtime path
- the package contains broader modules for Graphistry, telemetry, NCCL, and
  notebook tooling, but those should be treated as **capabilities in progress**
  rather than universally validated production surfaces
- when docs say a feature is available, that should mean the module and API are
  present in the package; when docs say a feature is validated, that should mean
  you have actually exercised it in your published notebooks or release process

## Troubleshooting

### Import succeeds but bootstrap is incomplete

If import works but runtime pieces are missing, check:

```python
import os
print(os.environ.get("LLAMA_SERVER_PATH"))
```

If that value is empty, re-run the install in a clean environment and confirm
that the machine has the GPU/runtime layout expected by the package.

### `detect_cuda()` reports no GPU

That usually means one of these:

- no NVIDIA GPU is attached
- drivers are not available in the current session
- you are not running in the Kaggle or Linux GPU environment the package expects

### OpenTelemetry imports fail

Install the telemetry extra:

```bash
pip install "llamatelemetry[telemetry] @ git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1"
```

### Graphistry imports fail

Install the graphistry extra:

```bash
pip install "llamatelemetry[graphistry] @ git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1"
```

### Kaggle notebook drift

In Kaggle, a restart after installation is sometimes the cleanest fix when a
notebook keeps references to an older package state.
