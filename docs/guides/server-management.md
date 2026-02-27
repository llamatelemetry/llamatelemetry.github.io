# Server Management Guide

`ServerManager` handles `llama-server` discovery, startup, health checks, and shutdown.

## Basic usage

```python
from llamatelemetry.server import ServerManager

server = ServerManager(server_url="http://127.0.0.1:8090")
server.start_server(model_path="/path/model.gguf", gpu_layers=99, ctx_size=4096)
```

## Discovery order (high level)

Server lookup checks:

1. `LLAMA_SERVER_PATH`
2. Package bootstrap locations
3. `LLAMA_CPP_DIR`
4. Cache locations (`~/.cache/llamatelemetry`)
5. Repository/dev paths
6. System `PATH`
7. Binary download fallback

## Startup options

Important `start_server(...)` params:

- `model_path`
- `host`, `port`
- `gpu_layers`
- `ctx_size`
- `n_parallel`
- `batch_size`, `ubatch_size`
- `flash_attn` (via extra kwargs mapping)
- `silent` for suppressing process output

## Health and lifecycle helpers

```python
healthy = server.check_server_health()
info = server.get_server_info()
server.restart_server(model_path="/path/model.gguf")
server.stop_server()
```

## Environment variables

- `LLAMA_SERVER_PATH`: direct executable override
- `LLAMA_CPP_DIR`: custom llama.cpp build root
- `LD_LIBRARY_PATH`: set internally for bundled libs when needed

## Operational tips

- Prefer explicit `server_url` and `port` consistency.
- Keep model path absolute when troubleshooting.
- If startup fails silently, rerun with `silent=False`.
