# Telemetry and Observability

`llamatelemetry.telemetry` integrates OpenTelemetry tracing and metrics with GPU-aware resource attributes. It can also export trace graphs to Graphistry.

## What you get

- Tracing spans for inference workflows
- GPU metrics and llama.cpp `/metrics` gauges
- OTLP export (gRPC/HTTP)
- Optional Graphistry trace visualization

## Basic setup

```python
from llamatelemetry.telemetry import setup_telemetry

tracer, meter = setup_telemetry(
    service_name="llamatelemetry-demo",
    service_version="0.1.0",
    otlp_endpoint="http://localhost:4317",
    enable_llama_metrics=True,
    llama_metrics_interval=5.0,
)
```

## Using with InferenceEngine

```python
import llamatelemetry as lt

engine = lt.InferenceEngine(
    enable_telemetry=True,
    telemetry_config={
        "service_name": "llamatelemetry-demo",
        "otlp_endpoint": "http://localhost:4317",
        "enable_llama_metrics": True,
    },
)
```

## Graphistry export

```python
tracer, meter = setup_telemetry(
    service_name="llamatelemetry-graphistry",
    enable_graphistry=True,
    graphistry_server="https://hub.graphistry.com",
)
```

## Kaggle secrets helper

```python
from llamatelemetry.telemetry import setup_otlp_env_from_kaggle_secrets

setup_otlp_env_from_kaggle_secrets(
    endpoint_key="OTLP_ENDPOINT",
    token_key="OTLP_TOKEN",
)
```

## Related reference

- [Telemetry API](../reference/telemetry-api.md)
- [Graphistry API](../reference/graphistry-api.md)
