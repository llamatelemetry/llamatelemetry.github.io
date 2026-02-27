# Get Started

This section gets you from zero to working inference quickly, then points to deeper guides.

## Paths

- Local machine quick setup:
  1. [Installation](installation.md)
  2. [Quickstart](quickstart.md)
- Kaggle setup (dual T4 focused):
  1. [Kaggle Quickstart](kaggle-quickstart.md)
  2. [Kaggle Environment guide](../guides/kaggle-environment.md)

## Runtime model

At a high level:

1. Install `llamatelemetry`.
2. Create `InferenceEngine`.
3. Load a model (local path, registry name, or `repo:file` syntax).
4. Inference calls go to `llama-server` over HTTP.
5. Optional telemetry records traces and metrics.

## First success checklist

- `engine.load_model(...)` completes.
- `engine.infer(...)` returns `result.success == True`.
- `result.text` contains generated output.
- `engine.get_metrics()` shows request and latency stats.

## Next

- Learn server controls: [Server Management](../guides/server-management.md)
- Learn models and registry: [Model Management](../guides/model-management.md)
- Learn endpoint-level control: [API Client](../guides/api-client.md)
