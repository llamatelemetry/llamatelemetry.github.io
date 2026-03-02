# Project File Map

This page maps the major folders and file groups in the `llamatelemetry` repository.

## Top-level layout

- `llamatelemetry/` — main Python package (core APIs, telemetry, Kaggle, Graphistry)
- `csrc/` — CUDA/C++ extension sources
- `core/` — tensor and device primitives
- `docs/` — mkdocs content for the code repo
- `examples/` — runnable example scripts
- `notebooks/` — 18 curated notebooks (Kaggle-ready)
- `tests/` — end-to-end and unit tests
- `scripts/` — release and HuggingFace helpers
- `releases/` — source and binary release artifacts

## Python package (`llamatelemetry/`)

- `__init__.py` — `InferenceEngine`, `InferResult`, bootstrap integration
- `server.py` — `ServerManager` lifecycle manager
- `models.py` — model registry, metadata, and loading helpers
- `gguf_parser.py` — GGUF file parser
- `api/` — OpenAI-compatible HTTP client + GGUF helpers
- `telemetry/` — OpenTelemetry tracing and GPU metrics
- `kaggle/` — Kaggle presets, secrets, GPU context
- `graphistry/` — Graphistry and RAPIDS integration
- `quantization/` — quantization helpers
- `unsloth/` — Unsloth model integration
- `cuda/` — CUDA graphs, tensor core utilities, Triton kernels
- `inference/` — batch inference, KV cache, FlashAttention helpers
- `louie/` — knowledge extraction and graph query client
- `chat.py`, `embeddings.py`, `jupyter.py`, `utils.py`

## CUDA/C++ sources (`csrc/`)

- `csrc/core/` — tensor and device primitives
- `csrc/ops/` — CUDA kernels (matmul)
- `csrc/bindings.cpp` — Python bindings

## Release artifacts (`releases/`)

See [Release Artifacts](release-artifacts.md) for a full breakdown of source and CUDA bundles.
