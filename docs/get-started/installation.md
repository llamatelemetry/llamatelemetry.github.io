# Installation

## Requirements

- Python `>=3.11`
- Recommended: NVIDIA GPU with CUDA support for high performance
- For optional features:
  - OpenTelemetry exporters
  - Graphistry + RAPIDS stack
  - Jupyter/ipywidgets

## Install from GitHub

```bash
pip install --no-cache-dir --force-reinstall \
  git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

## Optional dependency groups

`pyproject.toml` defines optional groups such as:

- `jupyter`
- `telemetry`
- `graphistry`
- `all`

Install extras from package distributions as needed, or install individual packages manually in notebook environments.

## Verify installation

```python
import llamatelemetry

print(llamatelemetry.__version__)  # expected: 0.1.0
print(llamatelemetry.check_cuda_available())
```

## Common setup notes

- `llamatelemetry` may bootstrap/download runtime binaries on first import.
- `LLAMA_SERVER_PATH` can override executable discovery.
- `LLAMA_CPP_DIR` can point to a custom llama.cpp build directory.

## Windows note

Some runtime/test paths in `v0.1.0` are Linux/Kaggle-oriented. If you run on Windows:

- Prefer UTF-8 terminal encoding.
- Expect some ecosystem features to be notebook/Linux-first.
- Use [Troubleshooting](../guides/troubleshooting.md) for known issues.
