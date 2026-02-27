# Quickstart (Local)

## Minimal flow

```python
from llamatelemetry import InferenceEngine

engine = InferenceEngine(server_url="http://127.0.0.1:8090")
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)

result = engine.infer("Give me 3 bullet points about quantization.")
print(result.text if result.success else result.error_message)
```

## Load model options

`InferenceEngine.load_model(model_name_or_path, ...)` supports:

- Registry model name:
  - `"gemma-3-1b-Q4_K_M"`
- Local file path:
  - `"/path/to/model.gguf"`
- Hugging Face format:
  - `"repo/id:file.gguf"`

## Essential inference options

```python
result = engine.infer(
    prompt="What is tensor parallelism?",
    max_tokens=128,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    stop_sequences=["\n\nUser:"],
)
```

## Metrics snapshot

```python
metrics = engine.get_metrics()
print(metrics["latency"])
print(metrics["throughput"])
```

## Cleanup

```python
engine.unload_model()
```

Or use context manager:

```python
from llamatelemetry import InferenceEngine

with InferenceEngine() as engine:
    engine.load_model("gemma-3-1b-Q4_K_M")
    print(engine.infer("Hello").text)
```

## Next

- [Server Management](../guides/server-management.md)
- [Model Management](../guides/model-management.md)
- [API Client guide](../guides/api-client.md)
