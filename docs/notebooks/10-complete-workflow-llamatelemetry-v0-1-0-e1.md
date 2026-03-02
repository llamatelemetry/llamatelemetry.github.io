# 10 Complete Workflow (Kaggle)

Source: `notebooks/10-complete-workflow-llamatelemetry-v0-1-0-e1.ipynb`


## Notebook focus

This page is a cell-by-cell walkthrough of the notebook, explaining the intent of each step and showing the exact code executed.


## Cell-by-cell walkthrough

### Cell 1 (Markdown)

# 10 Complete Workflow (Kaggle)

End-to-end pipeline: OTLP environment setup, server preset launch,
instrumented client, and inference — all in one notebook.

**What you will learn:**
- Load Grafana OTLP credentials from Kaggle Secrets
- Start llama-server from a preset
- Create an instrumented client with OpenTelemetry
- Run inference with full observability

**Requirements:** Kaggle T4 x2. Optional: Grafana Cloud OTLP secrets.

### Cell 2 (Markdown)

## 1) Install

### Cell 3 (Code)

**Summary:** Installs required dependencies and runtime tools.


```python
!pip -q install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

### Cell 4 (Markdown)

## 2) Load OTLP environment from Kaggle Secrets

Reads `GRAFANA_OTLP_ENDPOINT` and `GRAFANA_OTLP_HEADERS` (or `OTLP_ENDPOINT`
/ `OTLP_TOKEN`) from Kaggle Secrets and sets them as environment variables.

### Cell 5 (Code)

**Summary:** Imports core libraries: llamatelemetry.


```python
from llamatelemetry.kaggle.pipeline import (
    KagglePipelineConfig,
    load_grafana_otlp_env_from_kaggle,
    start_server_from_preset,
    setup_otel_and_client,
)
from llamatelemetry.kaggle import ServerPreset

otlp_env = load_grafana_otlp_env_from_kaggle()
print(f"OTLP configured: {bool(otlp_env)}")
```

### Cell 6 (Markdown)

## 3) Start llama-server from a preset

### Cell 7 (Code)

**Summary:** Works with GGUF models, quantization, or metadata.


```python
model_path = "/kaggle/input/your-model/model.gguf"

manager = start_server_from_preset(model_path, ServerPreset.KAGGLE_DUAL_T4)
print(f"Server started: {manager.check_server_health()}")
```

### Cell 8 (Markdown)

## 4) Setup OpenTelemetry + instrumented client

`setup_otel_and_client()` returns a dict with:
- `client` — `InstrumentedLlamaCppClient`
- `tracer` — OpenTelemetry tracer
- `meter` — OpenTelemetry meter

### Cell 9 (Code)

**Summary:** Sets up Graphistry for graph visualization or analytics.


```python
cfg = KagglePipelineConfig(
    enable_graphistry=False,
    enable_llama_metrics=True,
)

ctx = setup_otel_and_client("http://127.0.0.1:8080", cfg)
client = ctx["client"]
print(f"Pipeline context keys: {list(ctx.keys())}")
```

### Cell 10 (Markdown)

## 5) Run inference with telemetry

### Cell 11 (Code)

**Summary:** Works with GGUF models, quantization, or metadata.


```python
resp = client.chat_completions({
    "model": "local-gguf",
    "messages": [{"role": "user", "content": "Hello from the complete workflow!"}],
    "max_tokens": 64,
})
print(resp.choices[0].message.content)
```

### Cell 12 (Markdown)

## 6) Check server metrics

### Cell 13 (Code)

**Summary:** Imports core libraries: json.


```python
import json

health = manager.get_health()
print(json.dumps(health, indent=2, default=str))
```

### Cell 14 (Markdown)

## 7) Cleanup

### Cell 15 (Code)

**Summary:** Cleans up or shuts down running resources.


```python
manager.stop_server()
print("Done.")
```
