# Release Artifacts

`llamatelemetry` v0.1.0 ships as both source archives and CUDA binary bundles.
This page provides a complete breakdown of release contents, installation
methods, and artifact usage.

---

## Release Overview

| Property | Value |
|----------|-------|
| Version | v0.1.0 |
| Release date | 2026-02-02 |
| License | MIT |
| Python | >= 3.11 |
| CUDA | 12.x |
| Target GPU | Tesla T4 (SM 7.5) |
| Repository | [github.com/llamatelemetry/llamatelemetry](https://github.com/llamatelemetry/llamatelemetry) |

---

## Installation

### From GitHub (recommended)

```bash
pip install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

### From Source

```bash
git clone https://github.com/llamatelemetry/llamatelemetry.git
cd llamatelemetry
git checkout v0.1.0
pip install -e .
```

### With Optional Dependencies

```bash
# Telemetry support
pip install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0 \
    opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp

# Graphistry support
pip install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0 \
    pygraphistry pandas

# Full installation (all optional dependencies)
pip install -e ".[dev]"
```

### On Kaggle

In a Kaggle notebook cell:

```python
!pip install -q git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

---

## Source Archives

Files:

- `llamatelemetry-v0.1.0-source.tar.gz`
- `llamatelemetry-v0.1.0-source.zip`

Contents (top level):

```
llamatelemetry-v0.1.0/
  llamatelemetry/        Python package source (~40 files, 13K+ lines)
  csrc/                  C++/CUDA extension sources (7 files, ~650 lines)
  docs/                  MkDocs documentation content
  notebooks/             18 curated Kaggle-ready notebooks
  examples/              Runnable example scripts
  tests/                 Unit and end-to-end tests (246 tests)
  scripts/               Release and HuggingFace helper scripts
  pyproject.toml         Build configuration
  README.md              Project README
  LICENSE                MIT License
  CHANGELOG.md           Version history
```

These archives match the repository source tree exactly.

---

## CUDA Binary Bundle (Kaggle T4 x2)

File:

- `llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz`

This bundle contains pre-compiled CUDA 12 binaries targeting Tesla T4 GPUs
(SM 7.5, compute capability 7.5). Total size is approximately 961 MB.

### Binaries Included

| Binary | Purpose |
|--------|---------|
| `llama-server` | OpenAI-compatible HTTP server for GGUF inference |
| `llama-cli` | Command-line inference tool |
| `llama-bench` | Performance benchmarking tool |
| `llama-embedding` | Embedding generation tool |
| `llama-tokenize` | Tokenization utility |
| `llama-gguf` | GGUF file inspection tool |
| `llama-gguf-hash` | GGUF file hash verification |
| `llama-gguf-split` | GGUF file splitting utility |
| `llama-quantize` | Model quantization tool |
| `llama-imatrix` | Importance matrix generation |
| `llama-perplexity` | Perplexity evaluation tool |
| `llama-export-lora` | LoRA adapter export tool |
| `llama-cvector-generator` | Control vector generation tool |

### Libraries Included

| Library | Purpose |
|---------|---------|
| `lib/libnccl.so` | NVIDIA Collective Communications Library |
| `lib/libnccl.so.2` | NCCL versioned symlink |

### Scripts Included

| Script | Purpose |
|--------|---------|
| `start-server.sh` | Quick-start script for llama-server |
| `quantize.sh` | Model quantization helper script |

---

## Binary Bootstrap Process

When `llamatelemetry` is imported for the first time and the CUDA binaries are
not found locally, the bootstrap layer automatically downloads and extracts
the binary bundle.

```
import llamatelemetry
  --> Check for llama-server in llamatelemetry/binaries/cuda12/
  --> If missing: download llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz
  --> Extract to llamatelemetry/binaries/cuda12/
  --> Set LLAMA_SERVER_PATH environment variable
  --> Set LD_LIBRARY_PATH to include lib/ directory
```

The download URL is configured in `ServerManager._BINARY_BUNDLES` and points
to the GitHub Releases page:

```
https://github.com/llamatelemetry/llamatelemetry/releases/download/v0.1.0/
    llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz
```

### Cache Locations

The bootstrap layer checks these locations in order:

1. `llamatelemetry/binaries/cuda12/llama-server` (package directory)
2. `/usr/local/bin/llama-server` (system-wide)
3. `/usr/bin/llama-server` (system-wide)
4. `~/.cache/llamatelemetry/llama-server` (user cache)

---

## Kaggle Dataset Integration

For Kaggle notebooks, GGUF model files are typically uploaded as Kaggle Datasets
and attached to the notebook. The SDK's model registry and `KaggleEnvironment`
handle path resolution automatically.

```python
from llamatelemetry.kaggle import KaggleEnvironment

env = KaggleEnvironment.setup()
engine = env.create_engine("gemma-3-1b-Q4_K_M")
```

The `MODEL_REGISTRY` in `_internal/registry.py` contains 30+ curated models
with HuggingFace repo references, filenames, sizes, and VRAM requirements.

---

## Checksums

Each archive includes a `.sha256` file in `releases/v0.1.0/` for integrity
verification:

```bash
sha256sum -c llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz.sha256
sha256sum -c llamatelemetry-v0.1.0-source.tar.gz.sha256
```

---

## Release Directory Structure

```
releases/
  v0.1.0/
    llamatelemetry-v0.1.0-source.tar.gz
    llamatelemetry-v0.1.0-source.zip
    llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz
    *.sha256 checksum files
  v1.2.0/
    Binary archive + CUDA tar.gz (future release)
```

---

## Where to Go Next

- [Installation Guide](../get-started/installation.md) -- step-by-step setup
- [Kaggle Quickstart](../get-started/kaggle-quickstart.md) -- Kaggle-specific setup
- [Architecture Overview](architecture.md) -- how the SDK uses these artifacts
- [Changelog](changelog.md) -- version history
