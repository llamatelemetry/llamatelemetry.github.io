# Release Artifacts

`llamatelemetry` v0.1.0 ships as both source archives and CUDA binary bundles. This page summarizes the contents.

## Source archives

Files:

- `llamatelemetry-v0.1.0-source.tar.gz`
- `llamatelemetry-v0.1.0-source.zip`

Contents (top level):

- `llamatelemetry/` package source
- `docs/` documentation
- `notebooks/` curated notebook set
- `examples/`, `tests/`, `scripts/`
- `csrc/` CUDA/C++ sources

These match the repository source tree.

## CUDA binary bundle (Kaggle T4 x2)

File:

- `llamatelemetry-v0.1.0-cuda12-kaggle-t4x2.tar.gz`

Contents:

- `bin/llama-server` and llama.cpp CLI tools
- `lib/libnccl.so` and `lib/libnccl.so.2`
- `start-server.sh` and `quantize.sh`

Binaries included:

- `llama-server`
- `llama-cli`
- `llama-bench`
- `llama-embedding`
- `llama-tokenize`
- `llama-gguf`, `llama-gguf-hash`, `llama-gguf-split`
- `llama-quantize`, `llama-imatrix`
- `llama-perplexity`, `llama-export-lora`, `llama-cvector-generator`

## Checksums

Each archive includes a `.sha256` file in `releases/v0.1.0/` for integrity verification.

## How the SDK uses these

- The bootstrap layer downloads the CUDA bundle if the server binary is missing.
- The source archives are used for manual builds or offline installation.
