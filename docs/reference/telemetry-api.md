# Telemetry API Reference

## Module: `llamatelemetry.telemetry`

## Setup and capability

- `setup_telemetry(...)`
- `is_otel_available()`
- `is_graphistry_available()`
- `get_metrics_collector()`

## Auto instrumentation exports

- `instrument_inference`
- `inference_span`
- `batch_inference_span`
- `create_llm_attributes`
- `annotate_span_from_result`

## Client instrumentation exports

- `LlamaCppClientInstrumentor`
- `instrument_llamacpp_client`
- `uninstrument_llamacpp_client`

## Monitoring exports

- `PerformanceSnapshot`
- `InferenceRecord`
- `PerformanceMonitor`

## Supporting modules (advanced)

- `telemetry.tracer`
- `telemetry.metrics`
- `telemetry.monitor`
- `telemetry.resource`
- `telemetry.exporter`
- `telemetry.graphistry_export`

## Example

```python
from llamatelemetry.telemetry import setup_telemetry, inference_span

tracer, meter = setup_telemetry(service_name="demo", service_version="0.1.0")
with inference_span("demo-request", model="gemma", prompt="hello"):
    pass
```
