# Telemetry API Reference

The `llamatelemetry.telemetry` package provides OpenTelemetry-based distributed tracing and GPU metrics for LLM inference workloads. It implements the OpenTelemetry Gen AI semantic conventions with 45 `gen_ai.*` attributes and 5 histogram metrics, giving full observability into GGUF inference on CUDA hardware.

**Module:** `llamatelemetry.telemetry`

---

## Setup Functions

### setup_telemetry

```python
def setup_telemetry(
    service_name: str = "llamatelemetry",
    service_version: str = "0.1.1",
    otlp_endpoint: Optional[str] = None,
    otlp_headers: Optional[Dict[str, str]] = None,
    export_interval_ms: int = 5000,
    enable_console_export: bool = False,
    enable_graphistry: bool = False,
    graphistry_server: Optional[str] = None,
    enable_llama_metrics: bool = False,
    llama_metrics_interval: int = 15,
    resource_attributes: Optional[Dict[str, str]] = None,
) -> Tuple[TracerProvider, MeterProvider]
```

Initialize the OpenTelemetry `TracerProvider` and `MeterProvider` for llamatelemetry. Sets up OTLP exporters, resource attributes, and optional Graphistry trace export.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `service_name` | `str` | `"llamatelemetry"` | OTel service name, appears in trace UIs |
| `service_version` | `str` | `"0.1.1"` | Service version string |
| `otlp_endpoint` | `Optional[str]` | `None` | OTLP HTTP endpoint URL (e.g., `"https://otlp.example.com/v1/traces"`). If `None`, telemetry is collected but not exported over network. |
| `otlp_headers` | `Optional[Dict[str, str]]` | `None` | HTTP headers for authentication (e.g., `{"Authorization": "Bearer token"}`) |
| `export_interval_ms` | `int` | `5000` | Metrics export interval in milliseconds |
| `enable_console_export` | `bool` | `False` | Print traces/metrics to stdout (useful for debugging) |
| `enable_graphistry` | `bool` | `False` | Enable Graphistry trace visualization export |
| `graphistry_server` | `Optional[str]` | `None` | Graphistry hub URL if using a private hub |
| `enable_llama_metrics` | `bool` | `False` | Enable polling of llama-server `/metrics` endpoint for GPU and server-side metrics |
| `llama_metrics_interval` | `int` | `15` | Polling interval in seconds for llama-server metrics |
| `resource_attributes` | `Optional[Dict[str, str]]` | `None` | Additional OTel resource attributes (e.g., `{"deployment.environment": "kaggle"}`) |

**Returns:** `Tuple[TracerProvider, MeterProvider]` — the initialized providers.

```python
from llamatelemetry.telemetry import setup_telemetry

tracer_provider, meter_provider = setup_telemetry(
    service_name="my-inference-service",
    service_version="1.0.0",
    otlp_endpoint="https://otlp.example.com/v1/traces",
    otlp_headers={"Authorization": "Bearer my-token"},
    enable_llama_metrics=True,
    enable_graphistry=True,
)
```

---

### is_otel_available

```python
def is_otel_available() -> bool
```

Check whether the `opentelemetry-api` and `opentelemetry-sdk` packages are installed and importable. Returns `False` gracefully if not installed — llamatelemetry never raises `ImportError` for optional telemetry dependencies.

```python
from llamatelemetry.telemetry import is_otel_available

if is_otel_available():
    setup_telemetry(...)
else:
    print("OTel not installed — inference runs without tracing")
```

---

### is_graphistry_available

```python
def is_graphistry_available() -> bool
```

Check whether `pygraphistry` is installed. Returns `False` if not available, allowing the rest of the telemetry stack to operate without graph export.

```python
from llamatelemetry.telemetry import is_graphistry_available

if is_graphistry_available():
    setup_telemetry(enable_graphistry=True, graphistry_server="https://hub.graphistry.com")
```

---

### get_metrics_collector

```python
def get_metrics_collector() -> Optional[GpuMetricsCollector]
```

Return the global `GpuMetricsCollector` instance if telemetry has been initialized, or `None` otherwise.

```python
from llamatelemetry.telemetry import get_metrics_collector

collector = get_metrics_collector()
if collector:
    snapshot = collector.snapshot()
    print(f"GPU 0 util: {snapshot.gpu_utilization_pct[0]:.1f}%")
```

---

### setup_otlp_env_from_kaggle_secrets

```python
def setup_otlp_env_from_kaggle_secrets(
    endpoint_secret: str = "OTLP_ENDPOINT",
    token_secret: str = "OTLP_TOKEN",
    service_name_secret: str = "OTLP_SERVICE_NAME",
    fallback_service_name: str = "llamatelemetry",
) -> Dict[str, str]
```

Load OTLP connection settings from Kaggle secrets and set the corresponding `OTEL_*` environment variables. Returns the loaded values as a dictionary.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endpoint_secret` | `str` | `"OTLP_ENDPOINT"` | Kaggle secret name for OTLP HTTP endpoint |
| `token_secret` | `str` | `"OTLP_TOKEN"` | Kaggle secret name for bearer token |
| `service_name_secret` | `str` | `"OTLP_SERVICE_NAME"` | Kaggle secret name for service name |
| `fallback_service_name` | `str` | `"llamatelemetry"` | Default service name if secret is not set |

```python
from llamatelemetry.telemetry import setup_otlp_env_from_kaggle_secrets, setup_telemetry

# In a Kaggle notebook:
env = setup_otlp_env_from_kaggle_secrets()
setup_telemetry(
    service_name=env.get("service_name", "llamatelemetry"),
    otlp_endpoint=env.get("endpoint"),
    otlp_headers={"Authorization": f"Bearer {env.get('token', '')}"},
)
```

---

## Core Classes

### InferenceTracerProvider

A `TracerProvider` subclass that wraps the OTel SDK provider with GPU-aware resource detection and inference-specific span processors.

```python
from llamatelemetry.telemetry.tracer import InferenceTracerProvider

provider = InferenceTracerProvider(
    service_name="llamatelemetry",
    service_version="0.1.1",
    resource_attributes={"deployment.environment": "kaggle"},
)

tracer = provider.get_tracer("llamatelemetry.inference")

with tracer.start_as_current_span("llama.generate") as span:
    span.set_attribute("gen_ai.provider.name", "llamatelemetry")
    span.set_attribute("gen_ai.request.model", "gemma-3-4b-Q4_K_M.gguf")
    # ... run inference
    span.set_attribute("gen_ai.usage.output_tokens", 128)
```

**Key methods:**

| Method | Description |
|--------|-------------|
| `get_tracer(name)` | Return a named tracer from this provider |
| `add_span_processor(processor)` | Add a span processor (e.g., OTLP exporter) |
| `shutdown()` | Flush all pending spans and close exporters |
| `force_flush(timeout_ms)` | Flush all buffered spans within timeout |

---

### InferenceTracer

A helper that wraps `opentelemetry.trace.Tracer` to inject standard `gen_ai.*` attributes automatically on every inference span.

```python
from llamatelemetry.telemetry.tracer import InferenceTracer

tracer = InferenceTracer(
    model_name="gemma-3-4b-Q4_K_M.gguf",
    provider_name="llamatelemetry",
)

with tracer.inference_span(operation="chat", session_id="sess_abc") as span:
    # gen_ai.provider.name, gen_ai.request.model, gen_ai.operation.name
    # are automatically set on the span
    ...
    tracer.record_token_usage(span, input_tokens=145, output_tokens=287)
    tracer.record_timing(span, ttft_ms=125.0, tpot_ms=5.2)
```

**Key methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `inference_span` | `(operation, session_id, **attrs)` | Context manager that creates and auto-populates a `gen_ai.*` span |
| `record_token_usage` | `(span, input_tokens, output_tokens)` | Set `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` |
| `record_timing` | `(span, ttft_ms, tpot_ms)` | Add TTFT and TPOT events to the span |
| `record_finish_reason` | `(span, reasons)` | Set `gen_ai.response.finish_reasons` |

---

### GpuMetricsCollector

Collects GPU utilization, memory, temperature, and power metrics via `pynvml` (NVIDIA Management Library), and optionally polls llama-server's `/metrics` Prometheus endpoint.

```python
from llamatelemetry.telemetry.metrics import GpuMetricsCollector

collector = GpuMetricsCollector(
    meter=meter,                    # opentelemetry.metrics.Meter
    gpu_indices=[0, 1],             # Which GPUs to monitor
    llama_metrics_url=None,         # e.g. "http://127.0.0.1:8090/metrics"
    poll_interval_s=15,
)
collector.start()

# Later:
snapshot = collector.snapshot()
collector.stop()
```

**PerformanceSnapshot** — returned by `GpuMetricsCollector.snapshot()`:

```python
@dataclass
class PerformanceSnapshot:
    timestamp: float                   # Unix timestamp
    gpu_utilization_pct: List[float]   # Per-GPU utilization [0-100]
    gpu_memory_used_mb: List[float]    # Per-GPU memory used in MB
    gpu_memory_total_mb: List[float]   # Per-GPU total memory in MB
    gpu_temperature_c: List[float]     # Per-GPU temperature in °C
    gpu_power_w: List[float]           # Per-GPU power draw in Watts
    tokens_per_second: Optional[float] # From llama-server metrics (if enabled)
    prompt_tokens_per_second: Optional[float]  # Prefill throughput
    kv_cache_usage_pct: Optional[float]        # KV cache fill percentage
```

---

### PerformanceMonitor

High-level context manager that starts `GpuMetricsCollector`, collects snapshots during an inference session, and returns a summary report.

```python
from llamatelemetry.telemetry.monitor import PerformanceMonitor

with PerformanceMonitor(
    gpu_indices=[0, 1],
    llama_metrics_url="http://127.0.0.1:8090/metrics",
    poll_interval_s=5,
) as monitor:
    # run inference
    result = engine.infer("Hello, world!")

report = monitor.report()
print(f"Peak GPU 0 util: {report.peak_gpu_utilization[0]:.1f}%")
print(f"Avg tokens/sec: {report.avg_tokens_per_second:.1f}")
```

**PerformanceReport** fields:

| Field | Type | Description |
|-------|------|-------------|
| `duration_s` | `float` | Total monitoring duration |
| `n_snapshots` | `int` | Number of data points collected |
| `peak_gpu_utilization` | `List[float]` | Peak GPU% per device |
| `avg_gpu_utilization` | `List[float]` | Average GPU% per device |
| `peak_memory_mb` | `List[float]` | Peak VRAM used per device |
| `avg_tokens_per_second` | `Optional[float]` | Average decode throughput |
| `peak_tokens_per_second` | `Optional[float]` | Peak decode throughput |
| `avg_kv_cache_pct` | `Optional[float]` | Average KV cache utilization |

---

### InstrumentedLLMClient

An auto-instrumented wrapper around `LlamaCppClient` that automatically creates inference spans with all standard `gen_ai.*` attributes for every API call.

```python
from llamatelemetry.telemetry import InstrumentedLLMClient

client = InstrumentedLLMClient(
    base_url="http://127.0.0.1:8090",
    model_name="gemma-3-4b-Q4_K_M.gguf",
    tracer=tracer,      # opentelemetry.trace.Tracer
    meter=meter,        # opentelemetry.metrics.Meter
)

# Automatically traced — span includes gen_ai.* attrs and token counts
response = client.chat([{"role": "user", "content": "Hello!"}])
```

---

### LlamaCppClientInstrumentor

OTel-style instrumentor that patches `LlamaCppClient` to inject tracing without changing call sites.

```python
from llamatelemetry.telemetry.instrumentor import LlamaCppClientInstrumentor

LlamaCppClientInstrumentor().instrument()

# All subsequent LlamaCppClient calls are now traced automatically
from llamatelemetry.api import LlamaCppClient
client = LlamaCppClient("http://127.0.0.1:8090")
resp = client.chat.create(messages=[...])  # Span created automatically

LlamaCppClientInstrumentor().uninstrument()  # Remove instrumentation
```

---

### GraphistryTraceExporter

An OTel `SpanExporter` that writes inference traces to Graphistry as a live graph, enabling visual exploration of trace topology and token timings.

```python
from llamatelemetry.telemetry.graphistry_export import GraphistryTraceExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

exporter = GraphistryTraceExporter(
    graphistry_server="https://hub.graphistry.com",
    username="your-username",
    password="your-password",
    dataset_name="llamatelemetry-traces",
)

tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
```

---

## Semantic Conventions (semconv)

`llamatelemetry.telemetry.semconv` provides helpers for the 45 `gen_ai.*` OpenTelemetry semantic convention attributes.

### Helper Functions

#### set_gen_ai_attr

```python
def set_gen_ai_attr(span: Span, key: str, value: Any) -> None
```

Set a `gen_ai.*` span attribute by short name (without the `gen_ai.` prefix).

```python
from llamatelemetry.telemetry.semconv import set_gen_ai_attr

set_gen_ai_attr(span, "request.model", "gemma-3-4b-Q4_K_M.gguf")
set_gen_ai_attr(span, "usage.input_tokens", 145)
set_gen_ai_attr(span, "usage.output_tokens", 287)
```

#### set_gen_ai_provider

```python
def set_gen_ai_provider(span: Span, provider: str = "llamatelemetry") -> None
```

Set the `gen_ai.provider.name` attribute on a span.

```python
set_gen_ai_provider(span)                      # Sets "llamatelemetry"
set_gen_ai_provider(span, "openai")            # Sets "openai"
```

#### attr_name

```python
def attr_name(short_key: str) -> str
```

Return the full `gen_ai.*` attribute name from a short key.

```python
attr_name("request.model")   # → "gen_ai.request.model"
attr_name("usage.output_tokens")  # → "gen_ai.usage.output_tokens"
```

#### metric_name

```python
def metric_name(short_key: str) -> str
```

Return the full `gen_ai.*` metric name.

```python
metric_name("client.operation.duration")   # → "gen_ai.client.operation.duration"
metric_name("server.time_to_first_token")  # → "gen_ai.server.time_to_first_token"
```

---

## Gen AI Semantic Attributes (45 attributes)

All attributes are prefixed with `gen_ai.` and follow the OTel semantic conventions specification.

### Provider & Model

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.provider.name` | `str` | Provider identifier — use `"llamatelemetry"` for this SDK |
| `gen_ai.request.model` | `str` | Model name (e.g., `"gemma-3-4b-Q4_K_M.gguf"`) |
| `gen_ai.response.model` | `str` | Model that actually generated the response |

### Operation

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.operation.name` | `str` | Operation type: `"chat"`, `"text_completion"`, `"embeddings"` |
| `gen_ai.conversation.id` | `str` | Session or conversation ID for multi-turn tracking |

### Request Parameters

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.request.temperature` | `float` | Sampling temperature (0.0–2.0) |
| `gen_ai.request.top_p` | `float` | Top-P (nucleus sampling) parameter |
| `gen_ai.request.top_k` | `float` | Top-K sampling parameter |
| `gen_ai.request.max_tokens` | `int` | Maximum tokens to generate |
| `gen_ai.request.seed` | `int` | Random seed for reproducibility |
| `gen_ai.request.frequency_penalty` | `float` | Frequency penalty |
| `gen_ai.request.presence_penalty` | `float` | Presence penalty |
| `gen_ai.request.stop_sequences` | `List[str]` | Token sequences that stop generation |
| `gen_ai.request.choice.count` | `int` | Number of candidate completions |
| `gen_ai.request.encoding_formats` | `List[str]` | Encoding formats for embeddings |

### Token Usage

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.usage.input_tokens` | `int` | Prompt/input token count (preferred over deprecated `prompt_tokens`) |
| `gen_ai.usage.output_tokens` | `int` | Completion/output token count (preferred over deprecated `completion_tokens`) |
| `gen_ai.usage.cache_creation.input_tokens` | `int` | Tokens written to KV cache |
| `gen_ai.usage.cache_read.input_tokens` | `int` | Tokens read from KV cache |
| `gen_ai.token.type` | `str` | Token classification: `"input"` or `"output"` |

### Input & Output

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.input.messages` | `Any` | Chat history (JSON-serializable list) |
| `gen_ai.output.messages` | `Any` | Model responses (JSON-serializable list) |
| `gen_ai.output.type` | `str` | Output type: `"text"`, `"json"`, `"image"`, `"speech"` |
| `gen_ai.system_instructions` | `Any` | System prompts or instructions |
| `gen_ai.prompt.name` | `str` | Prompt template identifier |

### Response Metadata

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.response.id` | `str` | Unique completion ID from llama-server |
| `gen_ai.response.finish_reasons` | `List[str]` | Stop reasons: `["stop"]`, `["length"]` |

### Agent Management

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.agent.id` | `str` | Unique agent identifier |
| `gen_ai.agent.name` | `str` | Human-readable agent name |
| `gen_ai.agent.description` | `str` | Free-form agent description |
| `gen_ai.agent.version` | `str` | Agent version |

### Tools & Function Calling

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.tool.name` | `str` | Tool identifier |
| `gen_ai.tool.type` | `str` | Tool type: `"function"`, `"extension"`, `"retrieval"` |
| `gen_ai.tool.description` | `str` | Tool description |
| `gen_ai.tool.definitions` | `Any` | Available tool specifications (JSON) |
| `gen_ai.tool.call.id` | `str` | Tool invocation ID |
| `gen_ai.tool.call.arguments` | `Any` | Parameters passed to the tool |
| `gen_ai.tool.call.result` | `Any` | Tool execution output |

### RAG & Data Sources

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.data_source.id` | `str` | Data source identifier |
| `gen_ai.retrieval.query.text` | `str` | RAG retrieval query |
| `gen_ai.retrieval.documents` | `Any` | Retrieved documents with scores |

### Evaluation

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.evaluation.name` | `str` | Metric identifier (e.g., `"relevance"`) |
| `gen_ai.evaluation.score.value` | `float` | Numeric evaluation score |
| `gen_ai.evaluation.score.label` | `str` | Human-readable label |
| `gen_ai.evaluation.explanation` | `str` | Scoring rationale |

### Embeddings

| Attribute | Type | Description |
|-----------|------|-------------|
| `gen_ai.embeddings.dimension.count` | `int` | Output embedding dimensionality |

---

## Gen AI Metrics (5 histograms)

All metrics are OTel histograms registered on the `MeterProvider` initialized by `setup_telemetry()`.

| Metric Name | Unit | Description |
|-------------|------|-------------|
| `gen_ai.client.operation.duration` | `s` | End-to-end latency from client perspective (prefill + decode + network) |
| `gen_ai.client.token.usage` | `{token}` | Input and output token counts per operation |
| `gen_ai.server.request.duration` | `s` | Server-side generation time (time-to-last-byte) |
| `gen_ai.server.time_to_first_token` | `s` | Prefill latency — time until first token begins streaming |
| `gen_ai.server.time_per_output_token` | `s` | Decode step latency — reciprocal of tokens-per-second |

---

## Complete Telemetry Example

```python
import llamatelemetry
from llamatelemetry.telemetry import setup_telemetry, setup_otlp_env_from_kaggle_secrets
from llamatelemetry.telemetry.semconv import set_gen_ai_attr, set_gen_ai_provider
from llamatelemetry.telemetry.monitor import PerformanceMonitor
from opentelemetry import trace

# 1. Load OTLP config from Kaggle secrets
env = setup_otlp_env_from_kaggle_secrets()

# 2. Initialize telemetry
tracer_provider, meter_provider = setup_telemetry(
    service_name="kaggle-inference",
    otlp_endpoint=env.get("endpoint"),
    otlp_headers={"Authorization": f"Bearer {env.get('token', '')}"},
    enable_llama_metrics=True,
    llama_metrics_interval=10,
    enable_graphistry=True,
)

tracer = trace.get_tracer("llamatelemetry.example")

# 3. Monitor GPU and run traced inference
with PerformanceMonitor(gpu_indices=[0, 1], poll_interval_s=5) as monitor:
    with llamatelemetry.InferenceEngine(
        enable_telemetry=True,
        telemetry_config={
            "service_name": "kaggle-inference",
            "enable_llama_metrics": True,
        }
    ) as engine:
        engine.load_model("gemma-3-4b-Q4_K_M")

        with tracer.start_as_current_span("batch_inference") as span:
            set_gen_ai_provider(span)
            set_gen_ai_attr(span, "request.model", "gemma-3-4b-Q4_K_M.gguf")
            set_gen_ai_attr(span, "operation.name", "chat")
            set_gen_ai_attr(span, "request.max_tokens", 256)

            result = engine.infer("Explain OpenTelemetry in one paragraph.")

            set_gen_ai_attr(span, "usage.input_tokens", 12)
            set_gen_ai_attr(span, "usage.output_tokens", result.tokens_generated)
            set_gen_ai_attr(span, "response.finish_reasons", ["stop"])
            set_gen_ai_attr(span, "output.type", "text")

        print(result.text)

report = monitor.report()
print(f"Peak GPU 0 util: {report.peak_gpu_utilization[0]:.1f}%")
print(f"Avg decode throughput: {report.avg_tokens_per_second:.1f} tok/s")

tracer_provider.shutdown()
```

---

## Related Documentation

- [Guide: Telemetry and Observability](../guides/telemetry-observability.md)
- [Graphistry API](graphistry-api.md)
- [Kaggle API](kaggle-api.md) — `setup_otlp_env_from_kaggle_secrets`
- [Core API](core-api.md) — `InferenceEngine(enable_telemetry=True)`
