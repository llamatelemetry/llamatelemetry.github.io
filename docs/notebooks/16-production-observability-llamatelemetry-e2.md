# 16 Production Observability Pipeline

Source: `notebooks/16-production-observability-llamatelemetry-e2.ipynb`


## Notebook focus

This page is a cell-by-cell walkthrough of the notebook, explaining the intent of each step and showing the exact code executed.


## Cell-by-cell walkthrough

### Cell 1 (Markdown)

# 16 Production Observability Pipeline

Set up a production-grade observability pipeline on Kaggle using the
pipeline helpers.

**What you will learn:**
- Configure `KagglePipelineConfig` for production use
- Set up OTLP + instrumented client in one call
- Monitor service health alongside inference

**Requirements:** Kaggle T4 x2. Optional: Grafana Cloud credentials.

### Cell 2 (Markdown)

## 1) Install

### Cell 3 (Code)

**Summary:** Installs required dependencies and runtime tools.


```python
!pip -q install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
```

### Cell 4 (Markdown)

## 2) Configure the pipeline

`KagglePipelineConfig` controls service identity, OTLP endpoint,
Graphistry toggle, and llama metrics collection.

### Cell 5 (Code)

**Summary:** Imports core libraries: llamatelemetry. Sets up Graphistry for graph visualization or analytics.


```python
from llamatelemetry.kaggle.pipeline import (
    KagglePipelineConfig,
    load_grafana_otlp_env_from_kaggle,
    start_server_from_preset,
    setup_otel_and_client,
)
from llamatelemetry.kaggle import ServerPreset

cfg = KagglePipelineConfig(
    service_name="llamatelemetry-prod",
    service_version="0.1.0",
    enable_graphistry=False,
    enable_llama_metrics=True,
)
print(f"Config: {cfg}")
```

### Cell 6 (Markdown)

## 3) Load OTLP credentials

### Cell 7 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
otlp_env = load_grafana_otlp_env_from_kaggle()
print(f"OTLP endpoint set: {bool(otlp_env)}")
```

### Cell 8 (Markdown)

## 4) Start the server

### Cell 9 (Code)

**Summary:** Works with GGUF models, quantization, or metadata.


```python
model_path = "/kaggle/input/your-model/model.gguf"
manager = start_server_from_preset(model_path, ServerPreset.KAGGLE_DUAL_T4)
print(f"Server healthy: {manager.check_server_health()}")
```

### Cell 10 (Markdown)

## 5) Setup OTLP + instrumented client

### Cell 11 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
ctx = setup_otel_and_client("http://127.0.0.1:8080", cfg)
client = ctx["client"]
print(f"Pipeline keys: {list(ctx.keys())}")
```

### Cell 12 (Markdown)

## 6) Production inference loop

### Cell 13 (Code)

**Summary:** Works with GGUF models, quantization, or metadata.


```python
queries = [
    "What is CUDA?",
    "Explain llama.cpp architecture.",
    "How does tensor parallelism work?",
]

for query in queries:
    resp = client.chat_completions({
        "model": "local-gguf",
        "messages": [{"role": "user", "content": query}],
        "max_tokens": 48,
    })
    text = resp.choices[0].message.content
    print(f"Q: {query}")
    print(f"A: {text[:100]}...\n")
```

### Cell 14 (Markdown)

## 7) Health check

### Cell 15 (Code)

**Summary:** Imports core libraries: json. Fetches runtime metrics from llama-server or telemetry collectors.


```python
import json

health = manager.get_health()
metrics = manager.get_metrics()

print("Health:", json.dumps(health, indent=2, default=str))
if metrics:
    print(f"\nMetrics (first 500 chars):\n{metrics[:500]}")
```

### Cell 16 (Markdown)

## 8) Cleanup

### Cell 17 (Code)

**Summary:** Cleans up or shuts down running resources.


```python
manager.stop_server()
print("Done.")
```
