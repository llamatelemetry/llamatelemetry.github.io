# Telemetry API

The `llamatelemetry.telemetry` package provides OpenTelemetry tracing and metrics with GPU-aware resource attributes.

## Entry points

- `setup_telemetry(...)` — initialize tracer and meter
- `is_otel_available()` — check SDK availability
- `is_graphistry_available()` — check Graphistry availability
- `get_metrics_collector()` — access GPU metrics collector
- `setup_otlp_env_from_kaggle_secrets()` — load OTLP endpoints from Kaggle secrets

## Core components

- `InferenceTracerProvider` — tracer provider with export hooks
- `GpuMetricsCollector` — GPU + llama.cpp metrics collection
- `InferenceTracer` — helper for inference spans
- `GraphistryTraceExporter` — optional graph export

## Semantic conventions

`llamatelemetry.telemetry.semconv` provides helpers for GenAI semantic attributes:

- `set_gen_ai_attr()`
- `set_gen_ai_provider()`
- `attr_name()` and `metric_name()` helpers

## Related docs

- [Telemetry and Observability](../guides/telemetry-observability.md)
- [Graphistry API](graphistry-api.md)
