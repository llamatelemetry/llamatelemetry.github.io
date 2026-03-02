# Core API

## InferenceEngine

High-level interface for GGUF inference with `llama-server`.

**Constructor:**

```python
InferenceEngine(
    server_url="http://127.0.0.1:8090",
    enable_telemetry=False,
    telemetry_config=None,
)
```

**Key methods:**

- `check_for_updates()` — optional update check against GitHub
- `check_server()` — health probe for `/health`
- `load_model(model_name_or_path, **kwargs)` — load model with smart configuration
- `get_last_suitability_report()` — report from GGUF suitability check
- `infer(prompt, **kwargs)` — main inference call
- `generate(prompt, **kwargs)` — alias for `infer`
- `infer_stream(prompt, **kwargs)` — streaming inference
- `batch_infer(prompts, **kwargs)` — batch inference helper
- `get_metrics()` / `reset_metrics()` — basic in-process metrics
- `unload_model()` — stop and clean resources

## InferResult

Wrapper for inference results with convenience access:

- `success` — boolean
- `text` — generated output
- `tokens_generated` — output token count
- `latency_ms` — end-to-end latency
- `tokens_per_sec` — throughput
- `error_message` — error details if any

## Convenience functions

- `check_cuda_available()` — quick CUDA availability check
- `get_cuda_device_info()` — device info helper
- `quick_infer(prompt, model_name="...")` — one-shot inference helper

## Related modules

- [Server and Models](server-models.md)
- [Client API](client-api.md)
