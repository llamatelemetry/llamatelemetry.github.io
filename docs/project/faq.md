# FAQ

## Does `llamatelemetry` include its own model weights?

No. Models are loaded from local GGUF paths or downloaded from Hugging Face/registry references.

## Is this only for Kaggle?

No, but `v0.1.0` is strongly optimized for Kaggle dual T4 workflows.

## What is the fastest way to get started?

Use:

1. [Installation](../get-started/installation.md)
2. [Quickstart](../get-started/quickstart.md)

## When should I use `LlamaCppClient` instead of `InferenceEngine`?

Use `LlamaCppClient` when you need endpoint-level control (`slots`, `lora`, `props`, specialized sampling parameters, etc.).

## How do I enable telemetry?

Either:

- initialize `InferenceEngine(enable_telemetry=True, telemetry_config=...)`, or
- call `llamatelemetry.telemetry.setup_telemetry(...)` directly.

## Why are some features unavailable in my environment?

Many advanced capabilities are optional and depend on installed packages/hardware (`Triton`, OpenTelemetry exporters, Graphistry/RAPIDS, NVIDIA runtime).

## Where are notebooks documented?

See [Notebook Hub](../notebooks/index.md).

## How do I debug startup failures?

Start with [Troubleshooting](../guides/troubleshooting.md), especially server path checks and `silent=False` startup mode.
