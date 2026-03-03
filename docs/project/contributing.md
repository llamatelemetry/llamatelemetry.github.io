# Contributing to llamatelemetry

Thank you for your interest in contributing to `llamatelemetry`! This guide covers everything you need to get started: development setup, project structure, coding conventions, testing, documentation, and the pull request process.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Building the C++/CUDA Extension](#building-the-ccuda-extension)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Writing Documentation](#writing-documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Release Process](#release-process)

---

## Development Setup

### Prerequisites

- Python ≥ 3.11
- CUDA 12.x toolkit (for C++ extension development)
- CMake ≥ 3.24
- Git
- (Optional) A CUDA-capable GPU for running GPU-dependent tests

### Clone and install in editable mode

```bash
git clone https://github.com/llamatelemetry/llamatelemetry.git
cd llamatelemetry
pip install -e ".[dev]"
```

The `[dev]` extra installs all development dependencies:

```
pytest
pytest-cov
black
isort
mypy
mkdocs-material
zensical
```

### Install optional extras for full test coverage

```bash
# OpenTelemetry exporters
pip install opentelemetry-exporter-otlp-proto-http opentelemetry-semantic-conventions

# GPU metrics (NVIDIA)
pip install pynvml

# Graphistry visualization
pip install pygraphistry

# Unsloth fine-tuning support
pip install torch unsloth

# Jupyter widgets
pip install ipywidgets
```

### Verify the installation

```bash
python -c "import llamatelemetry; print(llamatelemetry.__version__)"
# Expected: 0.1.0
```

---

## Project Structure

```
llamatelemetry/              # Repository root
├── pyproject.toml           # Package metadata and build config
├── CMakeLists.txt           # CUDA/C++ extension build
├── mkdocs.yml               # Docs site config
├── requirements.txt         # Core runtime dependencies
├── requirements-jupyter.txt # Jupyter-specific extras
│
├── llamatelemetry/          # Python package source
│   ├── __init__.py          # InferenceEngine (main API)
│   ├── server.py            # ServerManager
│   ├── models.py            # Model registry + downloader
│   ├── chat.py              # ChatEngine, ConversationManager
│   ├── embeddings.py        # EmbeddingEngine, SemanticSearch
│   ├── jupyter.py           # Jupyter ChatWidget
│   ├── utils.py             # CUDA detection, config helpers
│   ├── gguf_parser.py       # GGUF binary parser
│   │
│   ├── api/                 # llama.cpp API wrappers
│   ├── telemetry/           # OpenTelemetry integration
│   ├── kaggle/              # Kaggle environment helpers
│   ├── inference/           # Flash attention, KV cache, batching
│   ├── cuda/                # CUDA graphs, Triton, TensorCore
│   ├── quantization/        # NF4, GGUF conversion, dynamic quant
│   ├── graphistry/          # Graph visualization
│   ├── louie/               # AI graph analysis
│   ├── unsloth/             # Fine-tuning and LoRA
│   └── _internal/           # Bootstrap and model registry
│
├── csrc/                    # C++/CUDA extension source
│   ├── bindings.cpp         # pybind11 module definition
│   ├── core/                # Device and Tensor classes
│   └── ops/                 # cuBLAS matmul ops
│
├── tests/                   # Test suite
├── docs/                    # Documentation source (MkDocs)
├── notebooks/               # Jupyter tutorial notebooks
├── examples/                # Standalone example scripts
└── scripts/                 # Build and utility scripts
```

---

## Building the C++/CUDA Extension

If you are modifying `csrc/`, you need to rebuild the extension:

```bash
# Build in-place (for development)
python setup.py build_ext --inplace

# Or use CMake directly
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cp llamatelemetry_cpp*.so ../llamatelemetry/
```

The extension targets CUDA SM 7.5 (Tesla T4) by default. To build for a different GPU:

```bash
cmake .. -DCUDA_ARCH=86   # For Ampere (A100, RTX 3090)
```

### Build requirements

- `pybind11 >= 2.10.0`
- CUDA toolkit 12.x
- GCC with C++17 support

---

## Running Tests

The test suite uses `pytest`. Run all tests:

```bash
pytest -q
# Expected: 246 passed, 24 skipped
```

Run with verbose output:

```bash
pytest -v
```

Run a specific test file:

```bash
pytest tests/test_llamatelemetry.py -v
```

Run tests with coverage:

```bash
pytest --cov=llamatelemetry --cov-report=term-missing
```

### Test categories

| File | Coverage |
|------|----------|
| `test_llamatelemetry.py` | Core imports, platform detection, GPU compatibility, binary download, server/engine lifecycle, metrics |
| `test_new_apis.py` | Quantization, Unsloth, CUDA graphs, inference APIs |
| `test_tensor_api.py` | C++ extension: Device, Tensor, matmul, memory management |
| `test_gguf_parser.py` | GGUF format parser correctness |
| `test_full_workflow.py` | End-to-end with a real model binary |
| `test_end_to_end.py` | End-to-end inference test |

### Skipped tests

Tests that require a live CUDA GPU, a downloaded model, or a running llama-server are automatically skipped if those resources are not available. This allows the test suite to run cleanly on CI without GPU hardware.

---

## Code Style

### Python

- **Formatter:** `black` (line length 100)
- **Import sorter:** `isort` (black-compatible profile)
- **Type hints:** Use type annotations on all public API methods

Format code before committing:

```bash
black llamatelemetry/ tests/
isort llamatelemetry/ tests/
```

Check types:

```bash
mypy llamatelemetry/ --ignore-missing-imports
```

### Key conventions

**Optional dependencies must never cause `ImportError`:**

```python
# Good — graceful degradation
try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False

def get_gpu_utilization():
    if not _PYNVML_AVAILABLE:
        return None
    # ... use pynvml
```

**Public APIs should be dataclasses or typed:**

```python
@dataclass
class InferResult:
    success: bool
    text: str
    tokens_generated: int
    latency_ms: float
    tokens_per_sec: float
    error_message: Optional[str] = None
```

**Context managers for resources:**

```python
# Always use context managers for ServerManager, NCCLCommunicator, etc.
with ServerManager(...) as server:
    # server is guaranteed to stop on exit
    pass
```

**Do not hardcode paths or model URLs.** Use `_internal.registry.MODEL_REGISTRY` and `_internal.bootstrap` for binary management.

### C++/CUDA

- Follow the existing style in `csrc/` (C++17, Google-ish style)
- CUDA kernels go in `.cu` files; headers in `.h` files
- All CUDA calls must check return values via `CUDA_CHECK(...)` macro
- RAII is preferred for GPU memory management (see `Tensor` class)

---

## Writing Documentation

Docs are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) using `zensical.toml` as the config.

### Build docs locally

```bash
pip install zensical mkdocs-material
mkdocs serve
# Open http://localhost:8000
```

### Build static site

```bash
mkdocs build
# Output in site/
```

### Documentation conventions

- **API Reference pages** — document every public class, method, and function with parameter tables and code examples
- **Guide pages** — tutorial-style with narrative explanation, code blocks, and use-case examples
- **Code blocks** — use `python` language tag; annotate complex blocks with `# (1)` and admonitions
- **No auto-generated API docs** — all reference docs are manually authored for clarity

### Where to add docs

| Content type | Location |
|-------------|----------|
| New public API | `docs/reference/` |
| New workflow or feature tutorial | `docs/guides/` |
| Installation or setup change | `docs/get-started/` |
| New notebook | `docs/notebooks/` |
| Architecture change | `docs/project/architecture.md` |

---

## Pull Request Process

1. **Fork** the repository on GitHub
2. **Create a branch** from `main` with a descriptive name:
   ```bash
   git checkout -b feat/add-awq-quantization
   git checkout -b fix/server-timeout-handling
   ```
3. **Write or update tests** for your changes
4. **Run the test suite** and ensure it passes:
   ```bash
   pytest -q
   ```
5. **Format your code:**
   ```bash
   black llamatelemetry/ tests/
   isort llamatelemetry/ tests/
   ```
6. **Update documentation** if your change affects public APIs or behavior
7. **Commit with a clear message** following the conventional commits style:
   ```
   feat: add AWQ quantization support
   fix: handle llama-server timeout on slow networks
   docs: add telemetry guide for Kaggle T4 setup
   refactor: extract GPU detection into multigpu module
   test: add coverage for NCCLCommunicator teardown
   ```
8. **Open a Pull Request** against the `main` branch with:
   - A summary of the change and motivation
   - Links to related issues
   - Notes on testing performed
   - Any breaking API changes

### PR checklist

- [ ] Tests pass (`pytest -q`)
- [ ] Code formatted (`black`, `isort`)
- [ ] Public APIs have type annotations
- [ ] Optional dependencies use graceful degradation
- [ ] Documentation updated or added for new features
- [ ] No large binaries or model files committed
- [ ] `CHANGELOG.md` updated in the source repo

---

## Issue Reporting

Please open issues at: [github.com/llamatelemetry/llamatelemetry/issues](https://github.com/llamatelemetry/llamatelemetry/issues)

**For bug reports, include:**

- llamatelemetry version (`python -c "import llamatelemetry; print(llamatelemetry.__version__)"`)
- Python version (`python --version`)
- CUDA version (`nvcc --version`)
- GPU model and driver version (`nvidia-smi`)
- Operating system and environment (Kaggle, Colab, local Linux)
- Minimal reproducing example
- Full traceback

**For feature requests:**

- Describe the use case and motivation
- Suggest an API design if applicable
- Note any optional dependencies the feature would require

---

## Release Process

Releases are managed by the core team. The general process:

1. Update version in `pyproject.toml` and `llamatelemetry/__init__.py`
2. Update `CHANGELOG.md` in the source repository
3. Build and test the CUDA binary bundle on Kaggle T4
4. Create a GitHub release with source and binary artifacts
5. Publish to PyPI via `twine`

**Release artifacts include:**

- Source distribution (`.tar.gz`, `.whl`) — Python-only
- CUDA binary archive — `llamatelemetry-vX.Y.Z-cudaX.X-t4-complete.tar.gz` with pre-compiled `llamatelemetry_cpp.so` and llama-server binary

---

## Community

- **GitHub Issues:** [github.com/llamatelemetry/llamatelemetry/issues](https://github.com/llamatelemetry/llamatelemetry/issues)
- **Discussions:** [github.com/llamatelemetry/llamatelemetry/discussions](https://github.com/llamatelemetry/llamatelemetry/discussions)

All contributors are expected to follow a respectful, inclusive code of conduct. Harassment or discrimination of any kind will not be tolerated.
