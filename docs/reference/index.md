# API Reference Index

This section documents the public `llamatelemetry` APIs and module surfaces. Use the pages below for detailed class and function references.

## Core

- [Core API](core-api.md) — `InferenceEngine`, `InferResult`, and core helpers
- [Server and Models](server-models.md) — `ServerManager`, model discovery, and registries
- [Client API](client-api.md) — `LlamaCppClient` and OpenAI-compatible interfaces

## Model and GGUF

- [GGUF API](gguf-api.md) — GGUF parsing, validation, and quantization helpers
- [Multi-GPU and NCCL](multigpu-nccl.md) — multi-GPU config and NCCL helpers

## Observability

- [Telemetry API](telemetry-api.md) — tracing, metrics, and OTLP export
- [Graphistry API](graphistry-api.md) — graph visualization utilities

## Environment and workflows

- [Kaggle API](kaggle-api.md) — Kaggle environment and presets
- [CUDA and Inference API](cuda-inference-api.md) — CUDA graphs, tensor cores, and inference optimizations
- [Quantization and Unsloth API](quantization-unsloth.md) — quantization helpers and Unsloth exports
- [Jupyter, Chat, and Embeddings API](jupyter-chat-embeddings.md) — notebook UX and embeddings
- [Louie API](louie-api.md) — knowledge extraction and NLP graph queries
