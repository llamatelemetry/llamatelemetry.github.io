# Examples Cookbook

This page collects practical mini-recipes you can copy into scripts or notebooks.

## 1) One-shot quick inference

```python
from llamatelemetry import quick_infer

text = quick_infer(
    prompt="Explain tokenization in one sentence.",
    model_path="/path/to/model.gguf",
    auto_start=True,
)
print(text)
```

## 2) Force local model path, no registry prompt

```python
from llamatelemetry import InferenceEngine

engine = InferenceEngine()
engine.load_model("/kaggle/working/models/model.gguf", interactive_download=False)
```

## 3) Native completion with custom sampling

```python
from llamatelemetry.api import LlamaCppClient

client = LlamaCppClient("http://127.0.0.1:8080")
resp = client.complete(
    prompt="Generate a concise CUDA optimization checklist:",
    n_predict=180,
    temperature=0.6,
    top_p=0.9,
    min_p=0.05,
    repeat_penalty=1.1,
)
print(resp.choices[0].text)
```

## 4) Structured chat response hint

```python
resp = client.chat.completions.create(
    messages=[{"role": "user", "content": "Return JSON with fields title and summary"}],
    response_format={"type": "json_object"},
    max_tokens=140,
)
print(resp.choices[0].message.content)
```

## 5) Model compatibility guard before download

```python
from llamatelemetry.models import SmartModelDownloader

downloader = SmartModelDownloader(vram_gb=15.0)
validation = downloader.validate_model("gemma-3-12b-Q4_K_M")
if validation["fits"]:
    path = downloader.download("gemma-3-12b-Q4_K_M")
else:
    print("Try alternatives:", validation["alternative_models"])
```

## 6) Telemetry-enabled engine

```python
from llamatelemetry import InferenceEngine

engine = InferenceEngine(
    enable_telemetry=True,
    telemetry_config={
        "service_name": "llm-demo",
        "service_version": "0.1.0",
        "otlp_endpoint": "http://localhost:4317",
    },
)
```

## 7) Kaggle one-liner + engine creation

```python
from llamatelemetry.kaggle import quick_setup

env = quick_setup()
engine = env.create_engine("gemma-3-1b-Q4_K_M")
```

## 8) GGUF validation script

```python
from llamatelemetry.api.gguf import validate_gguf, get_model_summary

ok, msg = validate_gguf("model.gguf")
print(ok, msg)
print(get_model_summary("model.gguf"))
```
