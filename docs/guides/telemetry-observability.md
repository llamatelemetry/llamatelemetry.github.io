---
title: Telemetry and Observability Guide
description: OpenTelemetry guide for llamatelemetry, including setup_telemetry, setup_grafana_otlp, instrumented clients, GPU metrics collection, and Kaggle secret loading.
---

# Telemetry and Observability

`llamatelemetry` includes an actual `llamatelemetry.telemetry` package in the
uploaded SDK snapshot, so this guide should describe what is clearly present in
code and avoid overclaiming what has been broadly validated.

The current package gives you four practical observability layers:

- OpenTelemetry tracer and meter setup
- GPU-aware metrics collection
- optional llama.cpp `/metrics` polling
- an instrumented client for `llama-server` style APIs

## What is available in the current package

The telemetry package exposes these notable entry points:

- `setup_telemetry()`
- `setup_grafana_otlp()`
- `get_metrics_collector()`
- `PerformanceMonitor`
- `InstrumentedLlamaCppClient`
- helper utilities for span annotation and auto-instrumentation

## 1. Minimal setup with `setup_telemetry()`

```python
from llamatelemetry.telemetry import setup_telemetry

tracer, meter = setup_telemetry(
    service_name="llamatelemetry-demo",
    service_version="0.1.1",
    otlp_endpoint="http://localhost:4317",
    llama_server_url="http://127.0.0.1:8080",
    enable_llama_metrics=True,
    llama_metrics_interval=5.0,
)

print(tracer)
print(meter)
```

If OpenTelemetry SDK packages are not installed, this function returns `(None,
None)` and warns instead of pretending setup succeeded.

## 2. Kaggle- or secret-driven OTLP setup

The package also exposes `setup_grafana_otlp()` and Kaggle secret helpers.

```python
from llamatelemetry.telemetry import setup_grafana_otlp

tracer, meter = setup_grafana_otlp(
    service_name="llamatelemetry-kaggle",
    service_version="0.1.1",
    enable_llama_metrics=True,
)
```

This is useful when you want to rely on OTLP-related environment variables or
Kaggle secrets rather than hard-coding the endpoint in notebook cells.

## 3. Enabling telemetry in `InferenceEngine`

At the high level, telemetry is wired into `InferenceEngine` itself.

```python
import llamatelemetry as lt

engine = lt.InferenceEngine(
    server_url="http://127.0.0.1:8080",
    enable_telemetry=True,
    telemetry_config={
        "service_name": "my-inference-service",
        "service_version": "0.1.1",
        "otlp_endpoint": "http://localhost:4317",
        "enable_llama_metrics": True,
        "llama_metrics_interval": 5.0,
    },
)
```

When you run `engine.infer(...)`, the engine can create spans and forward timing
and token information into the telemetry layer when the dependencies are
available.

## 4. Recording data with the instrumented client

The telemetry package includes `InstrumentedLlamaCppClient`.

```python
from llamatelemetry.telemetry import InstrumentedLlamaCppClient, setup_telemetry

tracer, meter = setup_telemetry(
    service_name="client-demo",
    otlp_endpoint="http://localhost:4317",
)

client = InstrumentedLlamaCppClient(base_url="http://127.0.0.1:8080")
```

This client is the better choice when you want a lower-level, telemetry-aware
HTTP client rather than the full `InferenceEngine` abstraction.

## 5. Lightweight local monitoring with `PerformanceMonitor`

Not every workflow needs OTLP export. The package also includes an in-process
monitor.

```python
from llamatelemetry.telemetry import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start()

monitor.record(latency_ms=120.0, tokens=64, success=True)
monitor.record(latency_ms=180.0, tokens=96, success=True)
monitor.record(latency_ms=0.0, tokens=0, success=False)

print(monitor.get_summary())
monitor.stop()
```

Use this for notebook experiments where you want fast local feedback without a
full observability stack behind it.

## 6. Access the active GPU metrics collector

```python
from llamatelemetry.telemetry import get_metrics_collector, setup_telemetry

setup_telemetry(service_name="metrics-demo")
collector = get_metrics_collector()
print(collector)
```

That collector is created during telemetry setup and is also used by some other
parts of the package such as the NCCL-oriented code paths.

## 7. Kaggle secret loading

There are two relevant paths in the current codebase.

### Telemetry-level helper

```python
from llamatelemetry.telemetry import setup_otlp_env_from_kaggle_secrets

print(setup_otlp_env_from_kaggle_secrets())
```

### Kaggle pipeline helper

```python
from llamatelemetry.kaggle import load_grafana_otlp_env_from_kaggle

print(load_grafana_otlp_env_from_kaggle())
```

The Kaggle helper supports the more Grafana-specific naming convention, while the
telemetry helper supports a more generic OTLP secret layout.

## 8. A practical end-to-end pattern

```python
import llamatelemetry as lt
from llamatelemetry.telemetry import setup_telemetry

tracer, meter = setup_telemetry(
    service_name="llamatelemetry-demo",
    otlp_endpoint="http://localhost:4317",
    llama_server_url="http://127.0.0.1:8080",
    enable_llama_metrics=True,
)

engine = lt.InferenceEngine(
    server_url="http://127.0.0.1:8080",
    enable_telemetry=True,
    telemetry_config={
        "service_name": "llamatelemetry-demo",
        "otlp_endpoint": "http://localhost:4317",
        "enable_llama_metrics": True,
    },
)
```

That pattern is intentionally simple: initialize telemetry, then let the engine
reuse the same observability direction.

## 9. Documentation guardrails for this section

This guide now avoids a few claims that were too aggressive:

- it does not hard-claim a specific count like “45 semantic attributes” unless
  you freeze that count with tests and release validation
- it does not imply every backend is equally verified just because OTLP export
  exists in code
- it distinguishes between **code present in the package** and **field-tested
  observability workflows**

## Suggested secret names

For generic OTLP use:

- `OTLP_ENDPOINT`
- `OTLP_TOKEN`

For Grafana-style Kaggle setups:

- `GRAFANA_OTLP_ENDPOINT`
- `GRAFANA_OTLP_HEADERS`
- `GRAFANA_OTLP_TOKEN`

## Related pages

- [Kaggle Environment](kaggle-environment.md)
- [Kaggle Quickstart](../get-started/kaggle-quickstart.md)
- [Quickstart](../get-started/quickstart.md)
