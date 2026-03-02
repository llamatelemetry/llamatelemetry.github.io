# Get Started Overview

This section gives you the fastest path to a working `llamatelemetry` setup, plus a mental model of how the SDK fits together.

## How the SDK is structured

`llamatelemetry` is a Python orchestration layer around `llama-server` (from `llama.cpp`). The package handles:

- Runtime bootstrap of CUDA binaries and shared libraries
- Model discovery (registry, HuggingFace downloads, or local paths)
- Server lifecycle management (`ServerManager`)
- Inference requests and response handling (`InferenceEngine`)
- Optional observability (OpenTelemetry traces and metrics)

## The shortest path to success

1. Install the SDK
2. Verify CUDA and GPU visibility
3. Load a small GGUF model
4. Run inference
5. (Optional) enable telemetry and metrics

## Choose your path

- **Local Linux / Workstation**: Use the standard [Installation](installation.md) + [Quickstart](quickstart.md).
- **Kaggle Dual T4**: Jump to [Kaggle Quickstart](kaggle-quickstart.md).
- **Observability Focus**: Start with [Telemetry and Observability](../guides/telemetry-observability.md) after Quickstart.
- **Model and GGUF Workflows**: Read [Model Management](../guides/model-management.md) and [GGUF API](../reference/gguf-api.md).

## What `llamatelemetry` does for you

- Finds or downloads the `llama-server` binary and sets `LD_LIBRARY_PATH`
- Downloads GGUF models from a curated registry with VRAM-aware recommendations
- Starts and manages `llama-server` with sensible defaults
- Exposes a clean inference API plus an OpenAI-compatible client
- Adds optional GPU-aware OpenTelemetry metrics and traces

## Next steps

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Kaggle Quickstart](kaggle-quickstart.md)
- [Notebook Hub](../notebooks/index.md)
