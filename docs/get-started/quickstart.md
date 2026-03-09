# Quickstart

This tutorial walks through every core feature of llamatelemetry from first
import to cleanup. By the end, you will have verified your GPU, loaded a model,
run single and batch inference, streamed tokens, used the low-level client API
for chat completions and embeddings, and inspected server metrics.

All code blocks are complete and copy-pasteable. They assume you have already
installed the SDK (see [Installation](installation.md)).

---

## 1. Import and verify CUDA

Start by confirming that the SDK is installed and your GPU is visible:

```python
import llamatelemetry as lt

# detect_cuda() probes the NVIDIA driver and returns GPU metadata
cuda_info = lt.detect_cuda()

print(f"CUDA available: {cuda_info['available']}")
print(f"CUDA version:   {cuda_info['version']}")

for gpu in cuda_info["gpus"]:
    print(f"  {gpu['name']} | {gpu['memory']} MB | "
          f"SM {gpu['compute_capability']} | "
          f"Driver {gpu['driver_version']}")
```

If `available` is `False`, see the
[troubleshooting section](installation.md#troubleshooting) in the installation
guide.

Optionally, call `setup_environment()` to configure `LLAMA_CPP_DIR`,
`LD_LIBRARY_PATH`, and `CUDA_VISIBLE_DEVICES`:

```python
lt.setup_environment()
```

---

## 2. Create an InferenceEngine

The `InferenceEngine` is the primary high-level API. It manages the llama-server
lifecycle and exposes inference methods:

```python
engine = lt.InferenceEngine(
    server_url="http://127.0.0.1:8080",  # default server address
    enable_telemetry=False,               # disable OTEL for now
)
```

The `server_url` parameter sets the address the engine uses to communicate with
the llama-server process. Port 8080 is the default. To enable OpenTelemetry
tracing and metrics, set `enable_telemetry=True` and provide a
`telemetry_config` dictionary (covered in the
[Telemetry Guide](../guides/telemetry-observability.md)).

You can also use `InferenceEngine` as a context manager for automatic cleanup:

```python
with lt.InferenceEngine(enable_telemetry=False) as engine:
    engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)
    result = engine.infer("Hello, world!")
    print(result.text)
# Server is automatically stopped when the context exits
```

---

## 3. Load a model

### From the built-in model registry

llamatelemetry includes a curated registry of 22+ GGUF models. Use a registry
name to download and load automatically:

```python
engine.load_model(
    "gemma-3-1b-Q4_K_M",   # registry name
    auto_start=True,         # start llama-server after loading
    auto_configure=True,     # auto-set gpu_layers, ctx_size based on GPU
    gpu_layers=None,         # None = auto-detect from VRAM
    ctx_size=None,           # None = auto-detect
    n_parallel=1,            # number of parallel request slots
    verbose=True,            # print loading progress
)
```

The `auto_configure=True` option inspects your GPU's VRAM and compute
capability to set appropriate values for `gpu_layers` and `ctx_size`. For a
Tesla T4 with 16 GB VRAM, a 1B Q4 model will offload all layers to GPU with a
large context window.

### From a Hugging Face repository

Specify a `repo:file` string to download a specific GGUF file from any
Hugging Face repository:

```python
engine.load_model(
    "bartowski/gemma-2-2b-it-GGUF:gemma-2-2b-it-Q4_K_M.gguf",
    auto_start=True,
)
```

If the repository is gated, set the `HF_TOKEN` environment variable before
loading.

### From a local file path

Point directly to a GGUF file on disk:

```python
engine.load_model(
    "/path/to/my-model-Q4_K_M.gguf",
    auto_start=True,
    gpu_layers=99,    # offload all layers to GPU
    ctx_size=4096,
)
```

### Controlling downloads

The `interactive_download` parameter (default `True`) prompts for confirmation
before downloading large files. Set it to `False` for non-interactive
environments:

```python
engine.load_model(
    "gemma-3-1b-Q4_K_M",
    auto_start=True,
    interactive_download=False,
    silent=True,       # suppress all output
)
```

The `report_suitability` parameter prints a summary of how well the model fits
your GPU's VRAM:

```python
engine.load_model("gemma-3-1b-Q4_K_M", report_suitability=True)
```

---

## 4. Run inference

### Single prompt

The `infer()` method sends a completion request to the running llama-server and
returns an `InferResult`:

```python
result = engine.infer(
    prompt="Explain what CUDA cores are in two sentences.",
    max_tokens=128,      # maximum tokens to generate
    temperature=0.7,     # sampling temperature
    top_p=0.9,           # nucleus sampling threshold
    top_k=40,            # top-k sampling
    seed=0,              # 0 = random seed
    stop_sequences=None, # optional list of stop strings
)
```

The `generate()` method is an alias for `infer()` with the same signature:

```python
result = engine.generate("What is llama.cpp?", max_tokens=64)
```

### Inspecting InferResult

The returned `InferResult` object contains everything you need:

```python
print(f"Success:          {result.success}")
print(f"Generated text:   {result.text}")
print(f"Tokens generated: {result.tokens_generated}")
print(f"Latency:          {result.latency_ms:.1f} ms")
print(f"Throughput:       {result.tokens_per_sec:.1f} tokens/sec")

if not result.success:
    print(f"Error: {result.error_message}")
```

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | Whether the request completed without error |
| `text` | `str` | The generated text |
| `tokens_generated` | `int` | Number of tokens in the response |
| `latency_ms` | `float` | End-to-end request latency in milliseconds |
| `tokens_per_sec` | `float` | Generation throughput |
| `error_message` | `str` or `None` | Error details if `success` is `False` |

### Controlling generation

Adjust sampling parameters to control output quality and diversity:

```python
# Deterministic output (greedy decoding)
result = engine.infer(
    "List three benefits of quantization.",
    max_tokens=256,
    temperature=0.0,
    top_k=1,
)

# Creative output
result = engine.infer(
    "Write a short poem about GPU computing.",
    max_tokens=200,
    temperature=1.2,
    top_p=0.95,
    top_k=100,
)

# Stop at specific sequences
result = engine.infer(
    "Q: What is GGUF?\nA:",
    max_tokens=128,
    stop_sequences=["\nQ:", "\n\n"],
)
```

---

## 5. Batch inference

Process multiple prompts in a single call:

```python
prompts = [
  "Explain tensor cores in one sentence.",
  "What is the GGUF file format?",
  "How does KV cache work in transformers?",
  "Describe continuous batching for LLM serving.",
]

results = engine.batch_infer(prompts, max_tokens=96)

for i, r in enumerate(results):
  print(f"\n--- Prompt {i + 1} ---")
  print(f"Text: {r.text}")
  print(f"Tokens/sec: {r.tokens_per_sec:.1f}")
```

Batch inference processes prompts sequentially through the server but provides a
convenient single-call API. For true concurrent serving, increase the
`n_parallel` parameter when loading the model.

---

## 6. Using the LlamaCppClient directly

For full control over the llama-server REST API, use `LlamaCppClient`. This
exposes the OpenAI-compatible endpoints as well as native llama.cpp endpoints:

```python
from llamatelemetry.api import LlamaCppClient

client = LlamaCppClient(base_url="http://127.0.0.1:8080")
```

Note that `LlamaCppClient` defaults to port 8080, while `InferenceEngine` and
`ServerManager` default to port 8080. When using the client with an engine, pass
the engine's server URL.

### Chat completion

```python
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is CUDA?"},
    ],
    max_tokens=200,
    temperature=0.7,
)

print(response.choices[0].message.content)
```

### Text completion

```python
response = client.complete(
  prompt="The three main benefits of model quantization are:",
  n_predict=128,
  temperature=0.5,
)

print(response.choices[0].text)
```

### Embeddings

```python
response = client.embeddings.create(
  input="GPU-accelerated inference with llamatelemetry",
)

embedding = response.data[0].embedding
print(f"Embedding dimension: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
```

### Tokenization

```python
tokens = client.tokenize("Hello, llamatelemetry!")

print(f"Token count: {len(tokens.tokens)}")
print(f"Token IDs: {tokens.tokens}")

text = client.detokenize(tokens.tokens)
print(f"Detokenized: {text}")
```

### Health check

```python
health = client.health()
print(f"Server status: {health}")
```

---

## 7. Streaming responses

For token-by-token output, use the streaming mode through the client:

```python
# Streaming chat completions
for chunk in client.chat.completions.create(
  messages=[{"role": "user", "content": "Explain flash attention step by step."}],
  max_tokens=256,
  stream=True,
):
  delta = chunk["choices"][0].get("delta", {})
  content = delta.get("content", "")
  print(content, end="", flush=True)

print()
```

Streaming requires the `sseclient-py` package:

```bash
pip install sseclient-py
```

---

## 8. Server metrics

Retrieve performance metrics from the running llama-server:

```python
metrics = engine.get_metrics()
print(metrics)
```

The metrics dictionary includes request counts, token throughput, queue depths,
and latency percentiles. For persistent metric collection with OpenTelemetry, see
the [Telemetry and Observability Guide](../guides/telemetry-observability.md).

---

## 9. Enable telemetry

To add OpenTelemetry instrumentation to your inference calls:

```python
engine = lt.InferenceEngine(
    enable_telemetry=True,
    telemetry_config={
        "service_name": "my-llm-service",
        "enable_llama_metrics": True,
    },
)


result = engine.infer("What is OpenTelemetry?", max_tokens=64)

# Traces and metrics are now being collected
# Configure OTLP export via environment variables:
#   OTEL_EXPORTER_OTLP_ENDPOINT=https://your-collector:4318
#   OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic ...
```

The telemetry module emits 45 `gen_ai.*` span attributes following the
OpenTelemetry Gen AI semantic conventions and 5 metrics instruments for latency,
throughput, and resource usage.

---

## 10. Cleanup

When you are done, unload the model and stop the server:

```python
engine.unload_model()
```

If you used the context manager pattern (`with lt.InferenceEngine(...) as
engine:`), cleanup happens automatically when the block exits.

---

## Complete working example

Here is a self-contained script that combines all the steps above:

```python
import llamatelemetry as lt

# 1. Check GPU
cuda_info = lt.detect_cuda()
assert cuda_info["available"], "No CUDA GPU found"
print(f"GPU: {cuda_info['gpus'][0]['name']}")

# 2. Set up environment
lt.setup_environment()

# 3. Create engine and load model
with lt.InferenceEngine(enable_telemetry=False) as engine:
    engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)

    # 4. Single inference
    result = engine.infer(
        "What are the advantages of GGUF over GGML?",
        max_tokens=128,
        temperature=0.7,
    )
    print(f"\n{result.text}")
    print(f"({result.tokens_generated} tokens, "
          f"{result.tokens_per_sec:.1f} tok/s, "
          f"{result.latency_ms:.0f} ms)")

    # 5. Batch inference
    prompts = [
        "Define quantization in one sentence.",
        "What is a KV cache?",
    ]
    for r in engine.batch_generate(prompts, max_tokens=64):
        print(f"\n{r.text}")

    # 6. Metrics
    print(f"\nServer metrics: {engine.get_metrics()}")

# Server stops automatically here
print("Done.")
```

---

## Next steps

- [Kaggle Quickstart](kaggle-quickstart.md) -- optimized workflow for Kaggle
  T4 x2 notebooks.
- [Model Management](../guides/model-management.md) -- registry details, VRAM
  budgets, and the 22+ curated models.
- [Server Management](../guides/server-management.md) -- port configuration,
  health checks, multi-slot serving.
- [Telemetry and Observability](../guides/telemetry-observability.md) --
  OpenTelemetry setup, Grafana dashboards, Gen AI attributes.
- [API Client Reference](../guides/api-client.md) -- full `LlamaCppClient` API
  surface.
- [Notebook Hub](../notebooks/index.md) -- 18 production-tested notebooks.
