# Examples Cookbook

The `examples/` directory provides runnable scripts that demonstrate end-to-end workflows.

## Example index

- `examples/api_usage_examples.py` — Quick usage patterns for quantization, CUDA optimizations, and API integration.
- `examples/complete_workflow_example.py` — End-to-end workflow from Unsloth to deployment.
- `examples/kaggle_split_gpu_observability.py` — Kaggle split-GPU inference with telemetry.

## Running an example

```bash
python examples/api_usage_examples.py
```

## What to look for

- How the engine bootstraps and loads models
- How multi-GPU presets are applied
- How telemetry and metrics are enabled
- How Graphistry/RAPIDS hooks are used for analytics

## Related docs

- [Inference Engine](inference-engine.md)
- [Telemetry and Observability](telemetry-observability.md)
- [Kaggle Environment](kaggle-environment.md)
