# API Reference Index

This reference documents the `llamatelemetry v0.1.0` Python SDK surface by module family.

## Reference sections

- [Core API](core-api.md)
- [Server and Models](server-models.md)
- [Client API](client-api.md)
- [GGUF API](gguf-api.md)
- [Multi-GPU and NCCL](multigpu-nccl.md)
- [Telemetry API](telemetry-api.md)
- [Kaggle API](kaggle-api.md)
- [Graphistry API](graphistry-api.md)
- [Quantization and Unsloth API](quantization-unsloth.md)
- [CUDA and Inference API](cuda-inference-api.md)
- [Jupyter, Chat, and Embeddings API](jupyter-chat-embeddings.md)
- [Louie API](louie-api.md)

## Version scope

This documentation targets:

- Package version: `0.1.0`
- Primary runtime pattern: llama.cpp server + Python orchestration

## Notes

- Some APIs are optional-dependency-gated.
- Some APIs are optimized for Kaggle/Linux environments.
- Endpoint behavior depends on `llama-server` version and launch flags.
