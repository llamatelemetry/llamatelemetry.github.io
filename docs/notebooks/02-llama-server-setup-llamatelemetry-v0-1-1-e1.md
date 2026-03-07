# 02 llama-server Setup and Introspection

Source: `notebooks/02-llama-server-setup-llamatelemetry-v0-1-1-e1.ipynb`


## Notebook focus

This page is a cell-by-cell walkthrough of the notebook, explaining the intent of each step and showing the exact code executed.


## Cell-by-cell walkthrough

### Cell 1 (Markdown)

# 02 llama-server Setup and Introspection

Deep-dive into `ServerManager` — the class that launches, monitors, and
introspects the llama-server binary.

**What you will learn:**
- Start llama-server with observability endpoints
- Query `/health`, `/props`, `/slots`, `/metrics`
- Inspect loaded model properties
- Gracefully stop and restart the server

**Requirements:** Kaggle notebook with GPU T4 x2 accelerator. A GGUF model
uploaded as a Kaggle dataset.

### Cell 2 (Markdown)

## 1) Install

### Cell 3 (Code)

**Summary:** Installs required dependencies and runtime tools.


```python
!pip -q install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1
```

### Cell 4 (Markdown)

## 2) Create a ServerManager

`ServerManager` defaults to `http://127.0.0.1:8080`. You can override the
port by passing a different `server_url`.

### Cell 5 (Code)

**Summary:** Imports core libraries: llamatelemetry. Configures and manages the llama-server process lifecycle. Works with GGUF models, quantization, or metadata.


```python
from llamatelemetry.server import ServerManager

manager = ServerManager(server_url="http://127.0.0.1:8080")

# Point this to your uploaded GGUF dataset
model_path = "/kaggle/input/your-model/model.gguf"
```

### Cell 6 (Markdown)

## 3) Start the server with observability endpoints

Key flags:
- `enable_metrics=True` — expose Prometheus metrics at `/metrics`
- `enable_props=True` — expose model properties at `/props`
- `enable_slots=True` — expose slot states at `/slots`
- `n_parallel` — number of concurrent inference slots

### Cell 7 (Code)

**Summary:** Starts or configures the llama-server backend.


```python
manager.start_server(
    model_path=model_path,
    gpu_layers=99,
    ctx_size=4096,
    n_parallel=2,
    batch_size=512,
    ubatch_size=128,
    enable_metrics=True,
    enable_props=True,
    enable_slots=True,
    verbose=True,
)

# Block until the server responds to /health
ready = manager.wait_ready(timeout=120)
print(f"Server ready: {ready}")
```

### Cell 8 (Markdown)

## 4) Query /health

### Cell 9 (Code)

**Summary:** Imports core libraries: json.


```python
import json

health = manager.get_health()
print(json.dumps(health, indent=2, default=str))
```

### Cell 10 (Markdown)

## 5) Query /props — loaded model properties

### Cell 11 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
props = manager.get_props()
print(json.dumps(props, indent=2, default=str) if isinstance(props, dict) else props)
```

### Cell 12 (Markdown)

## 6) Query /slots — inference slot states

### Cell 13 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
slots = manager.get_slots()
print(json.dumps(slots, indent=2, default=str) if isinstance(slots, list) else slots)
```

### Cell 14 (Markdown)

## 7) Query /metrics — Prometheus-format metrics

The `/metrics` endpoint returns Prometheus exposition format text.

### Cell 15 (Code)

**Summary:** Fetches runtime metrics from llama-server or telemetry collectors.


```python
metrics_text = manager.get_metrics()
if metrics_text:
    # Print first 1000 chars
    print(metrics_text[:1000])
else:
    print("No metrics available (enable_metrics may not be supported by this build)")
```

### Cell 16 (Markdown)

## 8) Server info and models

### Cell 17 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
info = manager.get_server_info()
print(json.dumps(info, indent=2, default=str))
```

### Cell 18 (Markdown)

## 9) Cleanup — stop the server

### Cell 19 (Code)

**Summary:** Cleans up or shuts down running resources.


```python
stopped = manager.stop_server()
print(f"Server stopped: {stopped}")
```
