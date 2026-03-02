# Server Management

`ServerManager` handles discovery, startup, health checks, and shutdown of the `llama-server` backend. It is used internally by `InferenceEngine` but is also available for direct control.

## Key capabilities

- Locate `llama-server` in standard paths or bootstrap cache
- Download a release bundle if the binary is missing
- Configure `LD_LIBRARY_PATH` for bundled CUDA libs
- Start/stop the server with custom parameters
- Health checks and readiness polling

## Basic usage

```python
from llamatelemetry.server import ServerManager

manager = ServerManager()
server_path = manager.find_llama_server()
print(server_path)
```

## Starting a server

```python
manager.start_server(
    model_path="/path/to/model.gguf",
    gpu_layers=40,
    ctx_size=2048,
    n_parallel=2,
    host="127.0.0.1",
    port=8090,
)
```

## Health and metrics

```python
print(manager.check_server_health())
print(manager.get_health())
print(manager.get_metrics())
print(manager.get_models())
```

## Shutdown

```python
manager.stop_server()
```

## Environment variables

- `LLAMA_SERVER_PATH` — explicit binary path
- `LLAMA_CPP_DIR` — custom llama.cpp build directory
- `LD_LIBRARY_PATH` — auto-populated to include bundled libs

## Related reference

- [Server and Models API](../reference/server-models.md)
- [Inference Engine Guide](inference-engine.md)
