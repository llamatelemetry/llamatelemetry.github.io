# FAQ

## What is `llamatelemetry`?

A CUDA-first Python SDK that orchestrates `llama-server` for GGUF inference with optional OpenTelemetry, Kaggle presets, and graph analytics.

## Does it run without a GPU?

Yes, but performance will be limited. The SDK is optimized for NVIDIA GPUs, especially Kaggle T4 x2.

## How does it download binaries?

On first import, the bootstrap layer checks for `llama-server` and downloads a CUDA bundle if needed. The binary is cached in the package directory or user cache.

## Where are models stored?

By default in `llamatelemetry/models/`. You can pass a local path or override via your own download logic.

## Is OpenTelemetry required?

No. Telemetry is optional. If the OTel SDK is not installed, the telemetry layer is disabled gracefully.

## How can I use my own llama.cpp build?

Set `LLAMA_CPP_DIR` or `LLAMA_SERVER_PATH` to point to your custom build.

## What environments are supported?

Linux is primary. Kaggle T4 x2 is the best-supported configuration. Windows support is limited.
