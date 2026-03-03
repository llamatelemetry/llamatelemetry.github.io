# Telemetry and Observability

llamatelemetry integrates OpenTelemetry tracing and metrics with GPU-aware resource attributes, providing full observability for LLM inference workloads. It exports to any OTLP-compatible backend (Grafana Cloud, Jaeger, Prometheus) and optionally visualizes traces as graphs in Graphistry.

## Overview

The telemetry module provides:

- **Tracing** -- OpenTelemetry spans for every inference call with 45 `gen_ai.*` semantic attributes
- **Metrics** -- 5 core inference metrics (latency, throughput, token counts, cache usage, errors)
- **OTLP Export** -- gRPC and HTTP protocol support for Grafana Cloud, Jaeger, and other backends
- **GPU Resource Attributes** -- automatic detection of GPU model, VRAM, driver version, CUDA version
- **Instrumented Client** -- `InstrumentedLlamaCppClient` auto-creates spans for every request
- **Graphistry Export** -- optional trace visualization as interactive graphs
- **PerformanceMonitor** -- lightweight in-process metrics aggregation

## Quick Setup

### setup_grafana_otlp()

The simplest way to get started with Grafana Cloud:

```python
from llamatelemetry.telemetry import setup_grafana_otlp

tracer, meter = setup_grafana_otlp()
```

This reads OTLP configuration from environment variables:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otlp-gateway-prod-us-east-0.grafana.net/otlp"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic <base64-encoded-credentials>"
```

### setup_telemetry()

For full control over telemetry configuration:

```python
from llamatelemetry.telemetry import setup_telemetry

tracer, meter = setup_telemetry(
    service_name="my-llm-service",
    service_version="0.1.0",
    otlp_endpoint="http://localhost:4317",
    enable_llama_metrics=True,
    llama_metrics_interval=5.0,
)
```

### Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `service_name` | `str` | `"llamatelemetry"` | OpenTelemetry service name |
| `service_version` | `str` | `"0.1.0"` | Service version tag |
| `otlp_endpoint` | `str` | `None` | OTLP collector endpoint |
| `otlp_protocol` | `str` | `"grpc"` | Protocol: `"grpc"` or `"http"` |
| `otlp_headers` | `dict` | `None` | Authentication headers |
| `enable_llama_metrics` | `bool` | `False` | Scrape llama-server `/metrics` |
| `llama_metrics_interval` | `float` | `5.0` | Metrics scrape interval (seconds) |
| `llama_metrics_url` | `str` | `None` | Override llama-server metrics URL |
| `enable_graphistry` | `bool` | `False` | Enable Graphistry trace export |
| `graphistry_server` | `str` | `None` | Graphistry server URL |

## Using with InferenceEngine

The simplest integration path is through `InferenceEngine`:

```python
import llamatelemetry as lt

engine = lt.InferenceEngine(
    server_url="http://127.0.0.1:8090",
    enable_telemetry=True,
    telemetry_config={
        "service_name": "my-inference-service",
        "service_version": "1.0.0",
        "otlp_endpoint": "http://localhost:4317",
        "enable_llama_metrics": True,
        "llama_metrics_interval": 5.0,
    },
)

with engine:
    engine.load_model("gemma-3-1b-Q4_K_M")
    result = engine.infer("What is CUDA?", max_tokens=128)
    # Span is automatically created with gen_ai.* attributes
```

## InstrumentedLlamaCppClient

For direct client usage with automatic telemetry:

```python
from llamatelemetry.telemetry import InstrumentedLlamaCppClient, setup_telemetry

# Initialize telemetry first
tracer, meter = setup_telemetry(
    service_name="my-client",
    otlp_endpoint="http://localhost:4317",
)

# Create instrumented client
client = InstrumentedLlamaCppClient(base_url="http://127.0.0.1:8090")

# Every call creates an OpenTelemetry span automatically
response = client.chat_completions({
    "messages": [{"role": "user", "content": "What is GGUF?"}],
    "max_tokens": 64,
    "temperature": 0.7,
})

print(response["choices"][0]["message"]["content"])
```

!!! warning "Method Name Difference"
    `InstrumentedLlamaCppClient` uses `chat_completions()` (plural) with a raw dict payload. This differs from `LlamaCppClient.chat_completion()` (singular) which takes keyword arguments.

## The 45 gen_ai.* Attributes

Every inference span includes semantic attributes following the OpenTelemetry Gen AI conventions:

### Request Attributes

| Attribute | Example Value | Description |
|-----------|---------------|-------------|
| `gen_ai.system` | `"llama.cpp"` | AI system identifier |
| `gen_ai.request.model` | `"gemma-3-1b"` | Model name |
| `gen_ai.request.max_tokens` | `128` | Max tokens requested |
| `gen_ai.request.temperature` | `0.7` | Sampling temperature |
| `gen_ai.request.top_p` | `0.9` | Nucleus sampling |
| `gen_ai.request.top_k` | `40` | Top-K sampling |
| `gen_ai.request.seed` | `42` | Random seed |
| `gen_ai.request.stop_sequences` | `["\\n"]` | Stop sequences |
| `gen_ai.request.frequency_penalty` | `0.0` | Frequency penalty |
| `gen_ai.request.presence_penalty` | `0.0` | Presence penalty |

### Response Attributes

| Attribute | Example Value | Description |
|-----------|---------------|-------------|
| `gen_ai.response.model` | `"gemma-3-1b-Q4_K_M"` | Actual model used |
| `gen_ai.response.id` | `"cmpl-abc123"` | Response ID |
| `gen_ai.response.finish_reasons` | `["stop"]` | Finish reason |
| `gen_ai.usage.input_tokens` | `15` | Prompt tokens |
| `gen_ai.usage.output_tokens` | `64` | Completion tokens |

### Performance Attributes

| Attribute | Example Value | Description |
|-----------|---------------|-------------|
| `gen_ai.server.latency_ms` | `342.5` | Server-side latency |
| `gen_ai.server.tokens_per_sec` | `186.7` | Token generation speed |
| `gen_ai.server.prompt_eval_ms` | `45.2` | Prompt evaluation time |
| `gen_ai.server.generation_ms` | `297.3` | Generation time |

### GPU Resource Attributes

| Attribute | Example Value | Description |
|-----------|---------------|-------------|
| `gpu.model` | `"Tesla T4"` | GPU model name |
| `gpu.vram_total_mb` | `15360` | Total VRAM in MB |
| `gpu.vram_used_mb` | `8192` | Used VRAM in MB |
| `gpu.driver_version` | `"535.129.03"` | NVIDIA driver version |
| `gpu.cuda_version` | `"12.2"` | CUDA runtime version |
| `gpu.compute_capability` | `"7.5"` | SM compute capability |

## The 5 Core Metrics

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `gen_ai.client.token.usage` | Histogram | tokens | Token count distribution (input + output) |
| `gen_ai.client.operation.duration` | Histogram | ms | End-to-end inference latency |
| `gen_ai.server.tokens_per_second` | Gauge | tok/s | Current generation throughput |
| `gen_ai.server.kv_cache_usage` | Gauge | ratio | KV cache utilization (0.0--1.0) |
| `gen_ai.client.error.count` | Counter | errors | Total inference errors |

## PerformanceMonitor

For lightweight in-process monitoring without OTLP export:

```python
from llamatelemetry.telemetry import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start()

# Record inference results
monitor.record(latency_ms=150.0, tokens=64, success=True)
monitor.record(latency_ms=200.0, tokens=128, success=True)
monitor.record(latency_ms=0.0, tokens=0, success=False)

# Get summary statistics
summary = monitor.get_summary()
print(f"Total requests: {summary['total_requests']}")
print(f"Success rate: {summary['success_rate']:.1%}")
print(f"Avg latency: {summary['avg_latency_ms']:.1f} ms")
print(f"P95 latency: {summary['p95_latency_ms']:.1f} ms")
print(f"Avg throughput: {summary['avg_tokens_per_sec']:.1f} tok/s")

# Export to DataFrame for analysis
df = monitor.records_to_dataframe()
print(df.describe())

monitor.stop()
```

## Grafana Cloud Integration

### Environment Variable Setup

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otlp-gateway-prod-us-east-0.grafana.net/otlp"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic <base64(instanceId:token)>"
export OTEL_SERVICE_NAME="llamatelemetry-prod"
```

### Code Setup

```python
from llamatelemetry.telemetry import setup_grafana_otlp

tracer, meter = setup_grafana_otlp()

# Use tracer for custom spans
with tracer.start_as_current_span("my-inference-pipeline") as span:
    span.set_attribute("pipeline.stage", "warmup")
    # ... inference code ...
```

## Kaggle Secrets for OTLP

On Kaggle, load OTLP credentials from notebook secrets:

```python
from llamatelemetry.kaggle.pipeline import load_grafana_otlp_env_from_kaggle

# Loads secrets and sets environment variables
load_grafana_otlp_env_from_kaggle()

# Then setup telemetry normally
from llamatelemetry.telemetry import setup_grafana_otlp
tracer, meter = setup_grafana_otlp()
```

Required Kaggle secrets:

| Secret Name | Value |
|-------------|-------|
| `GRAFANA_OTLP_ENDPOINT` | Grafana OTLP gateway URL |
| `GRAFANA_OTLP_TOKEN` | Base64-encoded `instanceId:token` |

## GraphistryTraceExporter

Export traces as interactive graph visualizations:

```python
from llamatelemetry.telemetry import setup_telemetry

tracer, meter = setup_telemetry(
    service_name="graph-demo",
    enable_graphistry=True,
    graphistry_server="https://hub.graphistry.com",
)

# Run inference -- traces are automatically exported to Graphistry
```

See the [Graphistry and RAPIDS](graphistry-rapids.md) guide for visualization details.

## Best Practices

- **Use `setup_grafana_otlp()`** for quick Grafana Cloud integration with minimal configuration.
- **Enable `llama_metrics`** to capture server-side KV cache and throughput metrics.
- **Set a unique `service_name`** per deployment to distinguish traces in your backend.
- **Use `PerformanceMonitor`** for local development when you do not need OTLP export.
- **Batch your OTLP exports** -- the default gRPC exporter batches automatically.
- **On Kaggle**, use `load_grafana_otlp_env_from_kaggle()` to avoid hardcoding credentials.

## Complete Example

```python
from llamatelemetry.telemetry import (
    setup_telemetry,
    InstrumentedLlamaCppClient,
    PerformanceMonitor,
)

# 1. Initialize telemetry
tracer, meter = setup_telemetry(
    service_name="demo-service",
    service_version="0.1.0",
    otlp_endpoint="http://localhost:4317",
    enable_llama_metrics=True,
)

# 2. Create instrumented client
client = InstrumentedLlamaCppClient(base_url="http://127.0.0.1:8090")

# 3. Start performance monitor
monitor = PerformanceMonitor()
monitor.start()

# 4. Run inference with automatic telemetry
prompts = ["What is CUDA?", "What is GGUF?", "What is NCCL?"]
for prompt in prompts:
    response = client.chat_completions({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
    })
    tokens = response.get("usage", {}).get("completion_tokens", 0)
    monitor.record(latency_ms=100.0, tokens=tokens, success=True)

# 5. Get summary
summary = monitor.get_summary()
print(f"Requests: {summary['total_requests']}")
print(f"Avg latency: {summary['avg_latency_ms']:.1f} ms")
monitor.stop()
```

## Related

- [Kaggle Environment](kaggle-environment.md) -- Kaggle OTLP secrets setup
- [Graphistry and RAPIDS](graphistry-rapids.md) -- trace graph visualization
- [API Client](api-client.md) -- uninstrumented vs. instrumented clients
- [Telemetry API Reference](../reference/telemetry-api.md)
