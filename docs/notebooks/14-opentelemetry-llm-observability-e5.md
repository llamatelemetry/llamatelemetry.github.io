# 14 OpenTelemetry LLM Observability

Source: `notebooks/14-opentelemetry-llm-observability-e5.ipynb`


## Notebook focus

This page is a cell-by-cell walkthrough of the notebook, explaining the intent of each step and showing the exact code executed.


## Cell-by-cell walkthrough

### Cell 1 (Markdown)

# 14 OpenTelemetry LLM Observability

Set up OTLP tracing and metrics with `setup_grafana_otlp()` and the
`InstrumentedLlamaCppClient`.

**What you will learn:**
- Initialize OpenTelemetry tracer and meter providers
- Create an instrumented client that emits gen_ai.* spans and metrics
- Run traced inference requests
- Inspect telemetry attributes

**Requirements:** Kaggle T4 x2 with a running llama-server. Optional:
Grafana Cloud OTLP endpoint for remote export.

### Cell 2 (Markdown)

## 1) Install

### Cell 3 (Code)

**Summary:** Installs required dependencies and runtime tools.


```python
!pip -q install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1
```

### Cell 4 (Markdown)

## 2) Initialize telemetry

`setup_grafana_otlp()` returns a `(tracer, meter)` tuple. It configures
OTLP exporters if `GRAFANA_OTLP_ENDPOINT` / `OTLP_ENDPOINT` is set in
the environment.

### Cell 5 (Code)

**Summary:** Imports core libraries: llamatelemetry.


```python
from llamatelemetry.telemetry import setup_grafana_otlp

tracer, meter = setup_grafana_otlp(
    service_name="llamatelemetry",
    service_version="0.1.1",
    llama_server_url="http://127.0.0.1:8090",
    enable_llama_metrics=True,
)
print(f"Tracer: {tracer}")
print(f"Meter:  {meter}")
```

### Cell 6 (Markdown)

## 3) Create an instrumented client

`InstrumentedLlamaCppClient` automatically creates spans and records
metrics for every inference call. It uses `chat_completions()` (plural)
which accepts a payload dict.

### Cell 7 (Code)

**Summary:** Imports core libraries: llamatelemetry. Initializes the OpenAI-compatible llama.cpp HTTP client.


```python
from llamatelemetry.telemetry.client import InstrumentedLlamaCppClient

client = InstrumentedLlamaCppClient(base_url="http://127.0.0.1:8090")
```

### Cell 8 (Markdown)

## 4) Run a traced inference

### Cell 9 (Code)

**Summary:** Works with GGUF models, quantization, or metadata.


```python
resp = client.chat_completions({
    "model": "local-gguf",
    "messages": [{"role": "user", "content": "What is OpenTelemetry?"}],
    "max_tokens": 64,
})
print(resp.choices[0].message.content)
```

### Cell 10 (Markdown)

## 5) Run a traced completion

### Cell 11 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
resp2 = client.completions({
    "prompt": "Explain OTLP in one sentence:",
    "max_tokens": 32,
})
print(resp2)
```

### Cell 12 (Markdown)

## 6) Traced embeddings

### Cell 13 (Code)

**Summary:** Works with GGUF models, quantization, or metadata.


```python
emb_resp = client.embeddings({
    "input": "Telemetry test",
    "model": "local-gguf",
})
print(f"Embedding dimensions: {len(emb_resp.data[0].embedding) if hasattr(emb_resp, 'data') else 'N/A'}")
```

### Cell 14 (Markdown)

## 7) Notes

- All spans are tagged with `gen_ai.*` semantic convention attributes
  (45 attributes defined in the SDK).
- 5 metrics are recorded: `gen_ai.client.token.usage`,
  `gen_ai.client.operation.duration`, etc.
- If OTLP endpoint is configured, spans and metrics are exported
  automatically to Grafana Cloud, Jaeger, or any OTLP-compatible backend.
