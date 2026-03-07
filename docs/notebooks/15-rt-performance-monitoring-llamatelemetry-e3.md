# 15 Real-Time Performance Monitoring

Source: `notebooks/15-rt-performance-monitoring-llamatelemetry-e3.ipynb`


## Notebook focus

This page is a cell-by-cell walkthrough of the notebook, explaining the intent of each step and showing the exact code executed.


## Cell-by-cell walkthrough

### Cell 1 (Markdown)

# 15 Real-Time Performance Monitoring

Use `PerformanceMonitor` to track inference latency, throughput, and
llama.cpp server metrics in real time.

**What you will learn:**
- Start/stop the `PerformanceMonitor`
- Record inference results and manual measurements
- Get performance summaries
- Pull and parse llama.cpp `/metrics` endpoint

**Requirements:** Kaggle T4 x2 with a running llama-server.

### Cell 2 (Markdown)

## 1) Install

### Cell 3 (Code)

**Summary:** Installs required dependencies and runtime tools.


```python
!pip -q install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1
```

### Cell 4 (Markdown)

## 2) Create and start the monitor

### Cell 5 (Code)

**Summary:** Imports core libraries: llamatelemetry.


```python
from llamatelemetry.telemetry import PerformanceMonitor

monitor = PerformanceMonitor(
    window_size=1000,
    snapshot_interval=5.0,
    collect_gpu_metrics=True,
)
monitor.start()
print("Monitor started.")
```

### Cell 6 (Markdown)

## 3) Run inference and record results

Use `monitor.record(result)` with an `InferResult` from the engine, or
`monitor.record_manual()` for custom measurements.

### Cell 7 (Code)

**Summary:** Imports core libraries: llamatelemetry. Creates or uses the high-level InferenceEngine to run GGUF inference. Loads a GGUF model (from registry, HF, or local path) and applies runtime settings. Runs inference and captures the generated output.


```python
import llamatelemetry as lt

engine = lt.InferenceEngine(enable_telemetry=False)
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)

# Run several inferences and record each
prompts = [
    "What is GPU computing?",
    "Explain CUDA in one sentence.",
    "What is tensor parallelism?",
]

for prompt in prompts:
    result = engine.generate(prompt, max_tokens=48)
    monitor.record(result)
    print(f"  {result.tokens_per_sec:.1f} tok/s | {result.latency_ms:.0f} ms")
```

### Cell 8 (Markdown)

## 4) Manual recording

### Cell 9 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
# Record a custom measurement (e.g., from an external benchmark)
monitor.record_manual(
    latency_ms=150.0,
    tokens_generated=32,
    success=True,
    model="custom-model",
)
print("Manual record added.")
```

### Cell 10 (Markdown)

## 5) Performance summary

### Cell 11 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
summary = monitor.get_summary()
print(summary)

# Pretty print
monitor.print_summary()
```

### Cell 12 (Markdown)

## 6) Pull llama.cpp /metrics

### Cell 13 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
monitor.record_metrics_from_llama_server(server_url="http://127.0.0.1:8080")
print("Server metrics recorded.")
```

### Cell 14 (Markdown)

## 7) Export to DataFrame

### Cell 15 (Code)

**Summary:** Executes notebook-specific logic or data processing for this step.


```python
try:
    df = monitor.records_to_dataframe()
    print(f"Records: {len(df)}")
    display(df)
except ImportError:
    print("pandas required for DataFrame export")
```

### Cell 16 (Markdown)

## 8) Cleanup

### Cell 17 (Code)

**Summary:** Cleans up or shuts down running resources.


```python
monitor.stop()
engine.unload_model()
print("Done.")
```
