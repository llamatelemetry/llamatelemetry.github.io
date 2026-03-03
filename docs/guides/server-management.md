# Server Management

`ServerManager` handles the full lifecycle of the `llama-server` backend process: binary discovery, bootstrapping, startup with GPU-aware configuration, health monitoring, metrics collection, and graceful shutdown. It is used internally by `InferenceEngine` but is also available for direct control.

## Overview

The server management layer provides:

- **Binary discovery** -- locates `llama-server` in standard paths, environment variables, or the bootstrap cache
- **Auto-bootstrap** -- downloads a pre-built release bundle (~961 MB) if no binary is found
- **GPU-aware startup** -- configures GPU layers, context size, batch parameters, and multi-GPU splits
- **Health monitoring** -- readiness polling, health checks, and llama.cpp metrics endpoint
- **Lifecycle management** -- start, stop, restart, and process supervision

## Creating a ServerManager

```python
from llamatelemetry.server import ServerManager

# Default -- targets http://127.0.0.1:8090
manager = ServerManager()

# Custom URL
manager = ServerManager(server_url="http://127.0.0.1:9000")
```

## Finding the llama-server Binary

Before starting a server, the manager must locate the `llama-server` binary:

```python
server_path = manager.find_llama_server()
if server_path:
    print(f"Found llama-server at: {server_path}")
else:
    print("Binary not found -- bootstrap will download it")
```

### Search Order

The manager searches for the binary in this order:

1. `LLAMA_SERVER_PATH` environment variable (explicit path)
2. `LLAMA_CPP_DIR` environment variable (custom build directory)
3. The llamatelemetry bootstrap cache (`~/.cache/llamatelemetry/`)
4. Common system paths (`/usr/local/bin`, etc.)

If no binary is found and `auto_start=True` is passed to `load_model()`, the bootstrap system downloads the appropriate release bundle.

## Starting a Server

```python
manager.start_server(
    model_path="/path/to/model.gguf",
    port=8090,
    host="127.0.0.1",
    gpu_layers=99,
    ctx_size=2048,
    n_parallel=1,
    batch_size=512,
    ubatch_size=128,
    enable_metrics=True,
    enable_props=True,
    enable_slots=True,
)
```

### Start Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | required | Path to the GGUF model file |
| `port` | `int` | `8090` | HTTP port for the server |
| `host` | `str` | `"127.0.0.1"` | Bind address |
| `gpu_layers` | `int` | `99` | Number of layers offloaded to GPU |
| `ctx_size` | `int` | `2048` | Context window size in tokens |
| `n_parallel` | `int` | `1` | Number of parallel inference slots |
| `batch_size` | `int` | `512` | Logical batch size |
| `ubatch_size` | `int` | `128` | Physical micro-batch size |
| `enable_metrics` | `bool` | `True` | Enable the `/metrics` Prometheus endpoint |
| `enable_props` | `bool` | `True` | Enable the `/props` endpoint |
| `enable_slots` | `bool` | `True` | Enable the `/slots` endpoint |
| `multi_gpu_config` | `MultiGPUConfig` | `None` | Multi-GPU configuration |
| `nccl_config` | `NCCLConfig` | `None` | NCCL communication config |

### Multi-GPU Startup

Pass a `MultiGPUConfig` for tensor-parallel or layer-split inference:

```python
from llamatelemetry.api.multigpu import MultiGPUConfig, SplitMode

multi_gpu = MultiGPUConfig(
    n_gpu_layers=-1,                   # All layers on GPU
    split_mode=SplitMode.LAYER,        # Layer-based split
    tensor_split=[0.5, 0.5],           # Equal split across 2 GPUs
    flash_attention=True,
)

manager.start_server(
    model_path="/path/to/model.gguf",
    multi_gpu_config=multi_gpu,
)
```

## Waiting for Readiness

After starting the server, wait until it finishes loading the model and is ready for requests:

```python
manager.start_server(model_path="/path/to/model.gguf")

# Block until the server responds to health checks
ready = manager.wait_ready(timeout=120)
if ready:
    print("Server is ready for inference")
else:
    print("Server failed to start within timeout")
```

The default timeout is generous enough for large models on Tesla T4 GPUs. The method polls the `/health` endpoint at regular intervals.

## Health and Monitoring

### Health Check

```python
# Simple boolean check
is_healthy = manager.check_server_health()
print(f"Healthy: {is_healthy}")

# Detailed health response
health = manager.get_health()
print(health)
# Example: {"status": "ok", "slots_idle": 1, "slots_processing": 0}
```

### Server Properties

```python
props = manager.get_props()
print(f"Model: {props.get('default_generation_settings', {}).get('model')}")
print(f"Context size: {props.get('default_generation_settings', {}).get('n_ctx')}")
```

### Slot Information

```python
slots = manager.get_slots()
for slot in slots:
    print(f"Slot {slot['id']}: state={slot['state']}, task={slot.get('task_id')}")
```

### Prometheus Metrics

```python
metrics_text = manager.get_metrics()
print(metrics_text)
# Returns Prometheus-format text with llama.cpp internal metrics:
# - prompt_tokens_total
# - generation_tokens_total
# - prompt_seconds_total
# - generation_seconds_total
# - kv_cache_usage_ratio
```

### Model Information

```python
models = manager.get_models()
for model in models:
    print(f"Model: {model.get('id')}")
```

### Server Info

```python
info = manager.get_server_info()
print(f"Version: {info.get('version')}")
print(f"Build: {info.get('build_info')}")
```

## Stopping and Restarting

### Graceful Shutdown

```python
manager.stop_server()
```

This sends a termination signal to the `llama-server` process and waits for it to exit. If the process does not exit within a timeout, it is forcefully killed.

### Restart

```python
manager.restart_server()
```

Restart is equivalent to `stop_server()` followed by `start_server()` with the same parameters. This is useful for reloading a model or recovering from a crashed server.

## Using with Presets

ServerManager works well with the Kaggle preset system:

```python
from llamatelemetry.kaggle.presets import get_preset_config, ServerPreset

preset = get_preset_config(ServerPreset.KAGGLE_DUAL_T4)
manager = ServerManager()

# Convert preset to start_server kwargs
manager.start_server(
    model_path="/path/to/model.gguf",
    **preset.to_server_kwargs(),
)
```

See the [Kaggle Environment](kaggle-environment.md) guide for more preset details.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LLAMA_SERVER_PATH` | Explicit path to the `llama-server` binary |
| `LLAMA_CPP_DIR` | Directory containing a custom llama.cpp build |
| `LD_LIBRARY_PATH` | Auto-populated by the bootstrap to include bundled CUDA libraries |

```bash
# Example: point to a custom build
export LLAMA_SERVER_PATH=/opt/llama.cpp/build/bin/llama-server
export LD_LIBRARY_PATH=/opt/llama.cpp/build/lib:$LD_LIBRARY_PATH
```

## Process Management Details

The `ServerManager` spawns `llama-server` as a subprocess. Key behaviors:

- **stdout/stderr** are captured and available for debugging
- The server process is terminated when the manager is garbage-collected
- If the Python process exits unexpectedly, the server may remain running -- use `lsof -i :8090` to find and kill orphaned processes
- On Kaggle notebooks, processes are automatically cleaned up when the session ends

## Best Practices

- **Use `InferenceEngine`** for most workflows -- it handles `ServerManager` internally.
- **Call `wait_ready()`** after `start_server()` to avoid race conditions.
- **Enable metrics** (`enable_metrics=True`) for production monitoring.
- **Use presets** on Kaggle to get optimal settings for T4 GPUs automatically.
- **Set `n_parallel > 1`** only if you need concurrent inference slots.
- **Check health periodically** in long-running services to detect crashes.

## Complete Example

```python
from llamatelemetry.server import ServerManager
from llamatelemetry.api.multigpu import MultiGPUConfig, SplitMode

manager = ServerManager(server_url="http://127.0.0.1:8090")

# Find or bootstrap the binary
server_path = manager.find_llama_server()
print(f"Using: {server_path}")

# Start with multi-GPU config
manager.start_server(
    model_path="/models/gemma-3-1b-Q4_K_M.gguf",
    gpu_layers=99,
    ctx_size=4096,
    n_parallel=2,
    enable_metrics=True,
)

# Wait for readiness
manager.wait_ready(timeout=120)

# Monitor
print(f"Health: {manager.check_server_health()}")
print(f"Slots: {manager.get_slots()}")
print(f"Metrics:\n{manager.get_metrics()}")

# Cleanup
manager.stop_server()
```

## Related

- [Inference Engine](inference-engine.md) -- high-level API that wraps ServerManager
- [Kaggle Environment](kaggle-environment.md) -- preset configurations
- [API Client](api-client.md) -- HTTP client for the running server
- [Server and Models API Reference](../reference/server-models.md)
