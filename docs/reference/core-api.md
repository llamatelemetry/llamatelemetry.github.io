# Core API Reference

## Module: `llamatelemetry`

## Classes

- `InferenceEngine`
- `InferResult`

## Functions

- `check_cuda_available() -> bool`
- `get_cuda_device_info() -> Optional[Dict[str, Any]]`
- `quick_infer(...) -> str`

## Utilities re-exported from `llamatelemetry.utils`

- `detect_cuda`
- `check_gpu_compatibility`
- `get_llama_cpp_cuda_path`
- `setup_environment`
- `find_gguf_models`
- `print_system_info`
- `load_config`
- `create_config_file`
- `get_recommended_gpu_layers`
- `validate_model_path`

## `InferenceEngine` primary methods

- `check_server()`
- `load_model(...)`
- `infer(...)`
- `infer_stream(...)`
- `batch_infer(...)`
- `get_metrics()`
- `reset_metrics()`
- `unload_model()`

## `InferResult` fields

- `success`
- `text`
- `tokens_generated`
- `latency_ms`
- `tokens_per_sec`
- `error_message`

## Example

```python
import llamatelemetry as lt

engine = lt.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)
result = engine.infer("What is OpenTelemetry?")
print(result.text if result.success else result.error_message)
```
