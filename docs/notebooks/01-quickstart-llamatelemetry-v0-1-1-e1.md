# 01 Quickstart: llamatelemetry v0.1.1

Source: `notebooks/01-quickstart-llamatelemetry-v0-1-1-e1.ipynb`


## Notebook focus

This page is a cell-by-cell walkthrough of the notebook, explaining the intent of each step and showing the exact code executed.


## Cell-by-cell walkthrough

### Cell 1 (Markdown)

# 01 Quickstart: llamatelemetry v0.1.1

Get up and running with GGUF inference on **Kaggle (dual T4)** in under 5 minutes.

**What you will learn:**
- Install the SDK from GitHub
- Detect available CUDA GPUs
- Load a small GGUF model via `InferenceEngine`
- Run inference and inspect results
- Use the OpenAI-compatible `LlamaCppClient`

**Requirements:** Kaggle notebook with GPU T4 x2 accelerator enabled.

### Cell 2 (Markdown)

## 1) Install llamatelemetry

### Cell 3 (Code)

**Summary:** Installs required dependencies and runtime tools.


```python
!pip -q install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1
```

### Cell 4 (Markdown)

## 2) Verify GPU availability

### Cell 5 (Code)

**Summary:** Imports core libraries: llamatelemetry.


```python
import llamatelemetry as lt
from llamatelemetry import detect_cuda

cuda_info = detect_cuda()
print(f"CUDA available: {cuda_info['available']}")
print(f"CUDA version:   {cuda_info.get('version')}")
for i, gpu in enumerate(cuda_info.get('gpus', [])):
    print(f"  GPU {i}: {gpu}")
```

### Cell 6 (Markdown)

## 3) Load a small GGUF model

`InferenceEngine` is the high-level API. It manages the llama-server binary,
downloads models from the built-in registry, and provides `generate()` / `infer()`.

### Cell 7 (Code)

**Summary:** Creates or uses the high-level InferenceEngine to run GGUF inference. Loads a GGUF model (from registry, HF, or local path) and applies runtime settings. Starts or configures the llama-server backend.


```python
engine = lt.InferenceEngine(enable_telemetry=False)

# Load a small model from the built-in registry.
# auto_start=True launches llama-server automatically.
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)

print(f"Model loaded: {engine.is_loaded}")
print(f"Server healthy: {engine.check_server()}")
```

### Cell 8 (Markdown)

## 4) Run inference

`generate()` (alias for `infer()`) returns an `InferResult` with:
- `success` — whether the request completed
- `text` — generated text
- `tokens_generated` — output token count
- `latency_ms` — end-to-end latency
- `tokens_per_sec` — throughput

### Cell 9 (Code)

**Summary:** Runs inference and captures the generated output.


```python
result = engine.generate("What is CUDA?", max_tokens=64)

print(f"Success:    {result.success}")
print(f"Tokens:     {result.tokens_generated}")
print(f"Latency:    {result.latency_ms:.1f} ms")
print(f"Throughput: {result.tokens_per_sec:.1f} tok/s")
print(f"\nOutput:\n{result.text}")
```

### Cell 10 (Markdown)

## 5) Batch inference

### Cell 11 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
prompts = [
    "Explain tensor cores in one sentence.",
    "What is llama.cpp?",
    "Define GGUF format briefly.",
]
results = engine.batch_infer(prompts, max_tokens=48)

for prompt, res in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {res.text[:120]}...\n")
```

### Cell 12 (Markdown)

## 6) Engine metrics

### Cell 13 (Code)

**Summary:** Imports core libraries: json. Fetches runtime metrics from llama-server or telemetry collectors.


```python
import json

metrics = engine.get_metrics()
print(json.dumps(metrics, indent=2, default=str))
```

### Cell 14 (Markdown)

## 7) Alternative: OpenAI-compatible LlamaCppClient

If you prefer the OpenAI chat-completions style API, use `LlamaCppClient` directly.
Note: `ServerManager` defaults to port **8080**, so pass the matching URL.

### Cell 15 (Code)

**Summary:** Imports core libraries: llamatelemetry. Initializes the OpenAI-compatible llama.cpp HTTP client. Works with GGUF models, quantization, or metadata.


```python
from llamatelemetry.api import LlamaCppClient

client = LlamaCppClient(base_url="http://127.0.0.1:8080")

# Convenience method (singular: chat_completion)
resp = client.chat_completion(
    messages=[{"role": "user", "content": "Hello from llama.cpp!"}],
    model="local-gguf",
    max_tokens=32,
)
print(resp.choices[0].message.content)
```

### Cell 16 (Code)

**Summary:** Works with GGUF models, quantization, or metadata.


```python
# OpenAI-style chained call
resp2 = client.chat.completions.create(
    messages=[{"role": "user", "content": "What is OpenTelemetry?"}],
    model="local-gguf",
    max_tokens=48,
)
print(resp2.choices[0].message.content)
```

### Cell 17 (Markdown)

## 8) Cleanup

### Cell 18 (Code)

**Summary:** Cleans up or shuts down running resources.


```python
engine.unload_model()
print("Done.")
```
