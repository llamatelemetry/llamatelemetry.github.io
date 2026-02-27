# Telemetry and Observability Guide

`llamatelemetry.telemetry` provides optional OpenTelemetry integration for traces and metrics.

## Setup

```python
from llamatelemetry.telemetry import setup_telemetry

tracer, meter = setup_telemetry(
    service_name="llamatelemetry-inference",
    service_version="0.1.0",
    otlp_endpoint="http://localhost:4317",
    enable_graphistry=False,
)
```

## Enable through `InferenceEngine`

```python
from llamatelemetry import InferenceEngine

engine = InferenceEngine(
    enable_telemetry=True,
    telemetry_config={
        "service_name": "my-llm-service",
        "service_version": "0.1.0",
        "otlp_endpoint": "http://localhost:4317",
    },
)
```

## What is collected

- Request-level inference span attributes
- Latency and token metrics
- Optional GPU resource metadata
- Optional export via OTLP and Graphistry adapters

## Auto instrumentation helpers

From `llamatelemetry.telemetry.auto_instrument`:

- `instrument_inference`
- `inference_span`
- `batch_inference_span`
- `create_llm_attributes`

From `llamatelemetry.telemetry.instrumentor`:

- `instrument_llamacpp_client`
- `uninstrument_llamacpp_client`

## Performance monitor

`PerformanceMonitor` offers polling-oriented runtime monitoring with snapshots and records.

## Production notes

- Keep telemetry optional in latency-sensitive inference paths.
- Validate exporter availability at startup.
- Test endpoint and credential configuration in staging before production deployment.
