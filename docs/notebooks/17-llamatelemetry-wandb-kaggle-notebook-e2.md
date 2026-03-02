# 17 LlamaTelemetry + Weights & Biases (Kaggle)

Source: `notebooks/17-llamatelemetry-wandb-kaggle-notebook-e2.ipynb`


## Notebook focus

This page is a cell-by-cell walkthrough of the notebook, explaining the intent of each step and showing the exact code executed.


## Cell-by-cell walkthrough

### Cell 1 (Markdown)

# 17 LlamaTelemetry + Weights & Biases (Kaggle)

Combine llama.cpp inference, OTLP telemetry, and W&B experiment tracking.

**What you will learn:**
- Load W&B and Grafana secrets from Kaggle
- Initialize a W&B run
- Start llama-server from a preset
- Run instrumented inference and log metrics to W&B
- Log server health data

**Requirements:** Kaggle T4 x2. Kaggle Secrets: `WANDB_API_KEY` (required),
`GRAFANA_OTLP_ENDPOINT` / `GRAFANA_OTLP_HEADERS` (optional).

### Cell 2 (Markdown)

## 1) Install

### Cell 3 (Code)

**Summary:** Installs required dependencies and runtime tools. Initializes Weights & Biases logging or experiment tracking.


```python
!pip -q install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.0
!pip -q install wandb
```

### Cell 4 (Markdown)

## 2) Load Kaggle Secrets

### Cell 5 (Code)

**Summary:** Sets or updates environment variables for configuration. Imports core libraries: os. Sets environment variables for runtime configuration.


```python
import os

try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    WANDB_API_KEY = secrets.get_secret('WANDB_API_KEY')
    GRAFANA_OTLP_ENDPOINT = secrets.get_secret('GRAFANA_OTLP_ENDPOINT')
    GRAFANA_OTLP_HEADERS = secrets.get_secret('GRAFANA_OTLP_HEADERS')
except Exception:
    WANDB_API_KEY = os.environ.get('WANDB_API_KEY')
    GRAFANA_OTLP_ENDPOINT = None
    GRAFANA_OTLP_HEADERS = None

print(f"WANDB_API_KEY set: {bool(WANDB_API_KEY)}")
print(f"OTLP endpoint set: {bool(GRAFANA_OTLP_ENDPOINT)}")
```

### Cell 6 (Markdown)

## 3) Initialize W&B

### Cell 7 (Code)

**Summary:** Sets or updates environment variables for configuration. Imports core libraries: wandb. Initializes Weights & Biases logging or experiment tracking. Sets environment variables for runtime configuration.


```python
if WANDB_API_KEY:
    os.environ['WANDB_API_KEY'] = WANDB_API_KEY

import wandb

run = wandb.init(
    project='llamatelemetry',
    name='kaggle-llama.cpp-e2',
    reinit=True,
    config={
        'sdk_version': '0.1.0',
        'preset': 'KAGGLE_DUAL_T4',
    },
)
print(f"W&B run: {run.name}")
```

### Cell 8 (Markdown)

## 4) Start llama-server

### Cell 9 (Code)

**Summary:** Imports core libraries: llamatelemetry. Works with GGUF models, quantization, or metadata.


```python
from llamatelemetry.kaggle import ServerPreset
from llamatelemetry.kaggle.pipeline import start_server_from_preset

model_path = '/kaggle/input/your-model/model.gguf'
manager = start_server_from_preset(model_path, ServerPreset.KAGGLE_DUAL_T4)
print(f"Server healthy: {manager.check_server_health()}")
```

### Cell 10 (Markdown)

## 5) Setup telemetry + instrumented client

### Cell 11 (Code)

**Summary:** Imports core libraries: llamatelemetry. Sets up Graphistry for graph visualization or analytics.


```python
from llamatelemetry.kaggle.pipeline import (
    KagglePipelineConfig,
    load_grafana_otlp_env_from_kaggle,
    setup_otel_and_client,
)

load_grafana_otlp_env_from_kaggle()

cfg = KagglePipelineConfig(
    enable_graphistry=False,
    enable_llama_metrics=True,
)
ctx = setup_otel_and_client('http://127.0.0.1:8080', cfg)
client = ctx['client']
print(f"Pipeline keys: {list(ctx.keys())}")
```

### Cell 12 (Markdown)

## 6) Run inference + log to W&B

### Cell 13 (Code)

**Summary:** Initializes Weights & Biases logging or experiment tracking. Works with GGUF models, quantization, or metadata.


```python
queries = [
    'Explain llama.cpp in one paragraph.',
    'What is CUDA?',
    'How does quantization reduce model size?',
]

for i, query in enumerate(queries):
    resp = client.chat_completions({
        'model': 'local-gguf',
        'messages': [{'role': 'user', 'content': query}],
        'max_tokens': 64,
    })
    text = resp.choices[0].message.content
    print(f"[{i+1}] {text[:100]}...")

    wandb.log({
        'query': query,
        'output_preview': text[:200],
        'response_words': len(text.split()),
        'step': i,
    })
```

### Cell 14 (Markdown)

## 7) Log server health to W&B

### Cell 15 (Code)

**Summary:** Initializes Weights & Biases logging or experiment tracking.


```python
health = manager.get_health()
props = manager.get_props()

wandb.log({
    'server_health': str(health),
    'model_path': props.get('model_path') if isinstance(props, dict) else None,
})
print(f"Health: {health}")
```

### Cell 16 (Markdown)

## 8) Finish W&B run + cleanup

### Cell 17 (Code)

**Summary:** Initializes Weights & Biases logging or experiment tracking. Cleans up or shuts down running resources.


```python
wandb.finish()
manager.stop_server()
print("Done.")
```
