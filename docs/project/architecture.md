# Architecture Overview

`llamatelemetry v0.1.0` is organized as a Python orchestration SDK around `llama-server` with optional telemetry and GPU analytics layers.

## Layers

1. Bootstrap/runtime environment setup
2. High-level inference facade (`InferenceEngine`)
3. Server lifecycle management (`ServerManager`)
4. Endpoint-level client (`LlamaCppClient`)
5. Model and GGUF tooling
6. Optional observability, graph analytics, and optimization modules

## Request flow

1. `InferenceEngine.load_model(...)` resolves and configures model.
2. `ServerManager.start_server(...)` launches or attaches to server.
3. `InferenceEngine.infer(...)` sends HTTP request to `/completion`.
4. Response maps into `InferResult`.
5. Optional telemetry annotates spans and records metrics.

## Optional subsystem families

- `telemetry`: OpenTelemetry setup/instrumentation
- `kaggle`: environment and preset orchestration
- `graphistry`: graph analytics integration
- `quantization`/`api.gguf`: model conversion and quantization helpers
- `cuda`/`inference`: advanced optimization utilities
- `unsloth`: fine-tune/export integration

## Design characteristics

- Pragmatic high-level API with extensible lower-level controls.
- Heavy optional dependency model (graceful degradation when unavailable).
- Strong Kaggle-oriented presets and workflows in `v0.1.0`.
