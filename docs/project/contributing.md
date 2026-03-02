# Contributing

Thanks for contributing to `llamatelemetry`.

## Development setup

```bash
git clone https://github.com/llamatelemetry/llamatelemetry.git
cd llamatelemetry
pip install -e .[dev]
```

## Tests

```bash
pytest -q
```

## Code structure

- Python package: `llamatelemetry/`
- CUDA/C++ sources: `csrc/`
- Docs: `docs/`
- Notebooks: `notebooks/`

## Guidelines

- Keep optional dependencies optional
- Prefer clear docstrings and examples
- Add tests for new APIs when possible
- Avoid committing large binaries or model files
