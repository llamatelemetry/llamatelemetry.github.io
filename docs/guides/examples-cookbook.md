# Examples Cookbook

This cookbook provides complete, copy-paste-ready recipes for common llamatelemetry workflows. Each recipe includes a title, description, and working code.

## Recipe 1: Quick Inference

The simplest way to run inference with a registry model:

```python
import llamatelemetry as lt

with lt.InferenceEngine() as engine:
    engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True, auto_configure=True)

    result = engine.infer("What is CUDA?", max_tokens=128, temperature=0.7)

    if result.success:
        print(result.text)
        print(f"\n--- {result.tokens_generated} tokens in {result.latency_ms:.0f} ms "
              f"({result.tokens_per_sec:.1f} tok/s) ---")
    else:
        print(f"Error: {result.error_message}")
```

## Recipe 2: Multi-GPU Layer Split

Split a larger model across two GPUs on Kaggle:

```python
import llamatelemetry as lt
from llamatelemetry.api.multigpu import MultiGPUConfig, SplitMode
from llamatelemetry.server import ServerManager

# Configure layer-based split across 2 GPUs
multi_gpu = MultiGPUConfig(
    n_gpu_layers=-1,
    split_mode=SplitMode.LAYER,
    tensor_split=[0.5, 0.5],
    ctx_size=4096,
    batch_size=512,
    ubatch_size=128,
    flash_attention=True,
)

manager = ServerManager(server_url="http://127.0.0.1:8080")
manager.start_server(
    model_path="/path/to/llama-3.1-8b-Q4_K_M.gguf",
    multi_gpu_config=multi_gpu,
    enable_metrics=True,
)
manager.wait_ready(timeout=120)

# Now use InferenceEngine or LlamaCppClient
engine = lt.InferenceEngine(server_url="http://127.0.0.1:8080")
result = engine.infer("Explain tensor parallelism.", max_tokens=128)
print(result.text)

manager.stop_server()
```

## Recipe 3: Batch Processing

Process many prompts efficiently:

```python
import llamatelemetry as lt

prompts = [
    "Summarize the concept of GPU memory hierarchy.",
    "What is warp divergence in CUDA?",
    "Explain the difference between FP16 and FP32.",
    "What is flash attention and why does it matter?",
    "Describe the GGUF file format.",
    "What is quantization in the context of LLMs?",
    "Explain KV cache and its memory implications.",
    "What is tensor parallelism vs pipeline parallelism?",
]

with lt.InferenceEngine() as engine:
    engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)

    results = engine.batch_infer(prompts, max_tokens=128, temperature=0.5)

    for i, result in enumerate(results):
        if result.success:
            print(f"\n{'='*60}")
            print(f"Q: {prompts[i]}")
            print(f"A: {result.text[:200]}...")
            print(f"   ({result.tokens_generated} tokens, {result.tokens_per_sec:.1f} tok/s)")
        else:
            print(f"\n[{i}] FAILED: {result.error_message}")

    # Summary metrics
    metrics = engine.get_metrics()
    print(f"\nTotal: {metrics['requests']} requests, "
          f"{metrics['total_tokens']} tokens, "
          f"{metrics['total_latency_ms']:.0f} ms")
```

## Recipe 4: Streaming Chat

Stream tokens to the console in real-time:

```python
import llamatelemetry as lt

with lt.InferenceEngine() as engine:
    engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)

    print("Assistant: ", end="", flush=True)

    def on_token(token):
        print(token, end="", flush=True)

    result = engine.infer_stream(
        prompt="Write a detailed explanation of how GPU tensor cores work.",
        callback=on_token,
        max_tokens=256,
        temperature=0.7,
    )

    print(f"\n\n--- Streamed {result.tokens_generated} tokens "
          f"in {result.latency_ms:.0f} ms ---")
```

## Recipe 5: Telemetry Pipeline with Grafana

Full observability pipeline exporting to Grafana Cloud:

```python
import os
import llamatelemetry as lt
from llamatelemetry.telemetry import (
    setup_telemetry,
    InstrumentedLlamaCppClient,
    PerformanceMonitor,
)

# Set OTLP credentials (or use environment variables)
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://otlp-gateway.grafana.net/otlp"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "Authorization=Basic <credentials>"

# Initialize telemetry
tracer, meter = setup_telemetry(
    service_name="cookbook-demo",
    service_version="0.1.1",
    otlp_endpoint=os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"],
    enable_llama_metrics=True,
    llama_metrics_interval=5.0,
)

# Create instrumented client
client = InstrumentedLlamaCppClient(base_url="http://127.0.0.1:8080")

# Start performance monitor
monitor = PerformanceMonitor()
monitor.start()

# Run inference with automatic tracing
prompts = [
    "What is CUDA?",
    "Explain NCCL collectives.",
    "What is GGUF format?",
]

for prompt in prompts:
    with tracer.start_as_current_span("inference-request") as span:
        span.set_attribute("prompt.text", prompt[:100])

        response = client.chat_completions({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
        })

        content = response["choices"][0]["message"]["content"]
        tokens = response.get("usage", {}).get("completion_tokens", 0)
        print(f"Q: {prompt}")
        print(f"A: {content[:100]}...\n")

        monitor.record(latency_ms=100.0, tokens=tokens, success=True)

# Print summary
summary = monitor.get_summary()
print(f"Requests: {summary['total_requests']}")
print(f"Success rate: {summary['success_rate']:.0%}")
print(f"Avg latency: {summary['avg_latency_ms']:.1f} ms")
monitor.stop()
```

## Recipe 6: GGUF Inspection and Quantization Analysis

Inspect model files and analyze quantization options:

```python
from llamatelemetry.api.gguf import (
    gguf_report,
    report_model_suitability,
    quantization_matrix,
    estimate_gguf_size,
    recommend_quant_for_kaggle,
    GGUFReader,
)

model_path = "/path/to/model.gguf"

# Full GGUF report
report = gguf_report(model_path)
print(f"Architecture: {report['architecture']}")
print(f"Parameters: {report['parameters']}")
print(f"Quantization: {report['quantization']}")
print(f"File size: {report['file_size_mb']:.1f} MB")

# Suitability for a specific GPU
suitability = report_model_suitability(model_path, vram_gb=16)
print(f"\nSuitable for T4: {suitability['suitable']}")
print(f"Estimated VRAM: {suitability['estimated_vram_gb']:.1f} GB")
print(f"Recommended layers: {suitability['recommended_gpu_layers']}")

# Quantization matrix -- compare all quant types
matrix = quantization_matrix(parameters_b=7)
for quant_type, info in matrix.items():
    print(f"{quant_type}: {info['estimated_size_gb']:.1f} GB, "
          f"quality={info['relative_quality']}")

# Size estimation
size = estimate_gguf_size(parameters_b=7, quant_type="Q4_K_M")
print(f"\n7B Q4_K_M estimated size: {size:.1f} GB")

# Kaggle-specific recommendation
rec = recommend_quant_for_kaggle(parameters_b=7, n_gpus=2)
print(f"Recommended for Kaggle: {rec}")

# Read GGUF tensors
with GGUFReader(model_path) as reader:
    print(f"\nMetadata keys: {list(reader.metadata.keys())[:10]}")
    print(f"Tensor count: {len(reader.tensors)}")
    for tensor in reader.tensors[:5]:
        print(f"  {tensor.name}: {tensor.shape} ({tensor.dtype})")
```

## Recipe 7: Knowledge Graph Extraction

Extract entities and relationships from text:

```python
from llamatelemetry.louie.knowledge import KnowledgeExtractor, EntityType, RelationType

extractor = KnowledgeExtractor()

text = """
NVIDIA developed CUDA in 2006 as a parallel computing platform.
The Tesla T4 GPU uses the Turing architecture with 2560 CUDA cores
and 16 GB of GDDR6 memory. Flash attention, developed by Tri Dao
at Stanford, significantly reduces the memory footprint of the
attention mechanism in transformer models. llama.cpp by Georgi Gerganov
enables efficient CPU and GPU inference of large language models
using the GGUF format.
"""

kg = extractor.extract(text)

print("Entities:")
for entity in kg.entities:
    print(f"  [{entity.type.value}] {entity.name}")

print("\nRelationships:")
for rel in kg.relationships:
    print(f"  {rel.source} --[{rel.type.value}]--> {rel.target}")

# Filter by entity type
orgs = [e for e in kg.entities if e.type == EntityType.ORGANIZATION]
print(f"\nOrganizations: {[e.name for e in orgs]}")

techs = [e for e in kg.entities if e.type == EntityType.TECHNOLOGY]
print(f"Technologies: {[e.name for e in techs]}")
```

## Recipe 8: Embedding Search

Generate embeddings and find similar texts:

```python
from llamatelemetry.api import LlamaCppClient
import numpy as np

client = LlamaCppClient(base_url="http://127.0.0.1:8080")

# Corpus of documents
documents = [
    "CUDA is a parallel computing platform by NVIDIA.",
    "Flash attention reduces memory usage in transformers.",
    "GGUF is a file format for quantized language models.",
    "Tensor cores accelerate matrix multiplication on GPUs.",
    "NCCL provides GPU-to-GPU collective communications.",
    "KV cache stores key-value pairs during autoregressive generation.",
    "Quantization reduces model size by using lower-precision numbers.",
    "LoRA adapters enable parameter-efficient fine-tuning.",
]

# Generate embeddings for all documents
doc_embeddings = []
for doc in documents:
    emb = client.embed(doc)
    doc_embeddings.append(emb)

doc_embeddings = np.array(doc_embeddings)

# Search function
def search(query, top_k=3):
    query_emb = np.array(client.embed(query))
    # Cosine similarity
    similarities = np.dot(doc_embeddings, query_emb) / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_emb)
    )
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(documents[i], similarities[i]) for i in top_indices]

# Run a search
query = "How can I make model inference faster?"
results = search(query)

print(f"Query: {query}\n")
for doc, score in results:
    print(f"  [{score:.3f}] {doc}")
```

## Recipe 9: Complete Kaggle Pipeline

End-to-end pipeline for a Kaggle notebook with dual T4 GPUs:

```python
# Cell 1: Install (run once)
# !pip install git+https://github.com/llamatelemetry/llamatelemetry.git@v0.1.1

# Cell 2: Full pipeline
import llamatelemetry as lt
from llamatelemetry.kaggle.environment import KaggleEnvironment
from llamatelemetry.kaggle.presets import get_preset_config, ServerPreset
from llamatelemetry.kaggle.pipeline import (
    load_grafana_otlp_env_from_kaggle,
    setup_otel_and_client,
)
from llamatelemetry.kaggle.gpu_context import split_gpu_session

# Detect environment
env = KaggleEnvironment()
print(f"Runtime: {env.gpu_count}x {env.gpu_model}, {env.total_vram_gb} GB VRAM")

# Choose preset based on detected GPUs
if env.gpu_count >= 2:
    preset_name = ServerPreset.KAGGLE_DUAL_T4
else:
    preset_name = ServerPreset.KAGGLE_SINGLE_T4

preset = get_preset_config(preset_name)

# Load OTLP credentials from Kaggle secrets
load_grafana_otlp_env_from_kaggle()

# Split GPUs: inference on GPU 0, analytics on GPU 1
with split_gpu_session(llm_gpu=0, graph_gpu=1):
    # Create engine with telemetry
    with lt.InferenceEngine(
        enable_telemetry=True,
        telemetry_config={"service_name": "kaggle-pipeline"},
    ) as engine:
        engine.load_model(
            "gemma-3-1b-Q4_K_M",
            auto_start=True,
            **preset.to_load_kwargs(),
        )

        # Interactive inference
        questions = [
            "What is flash attention?",
            "How does KV cache work?",
            "What is model quantization?",
        ]

        for q in questions:
            result = engine.infer(q, max_tokens=128, temperature=0.7)
            print(f"\nQ: {q}")
            print(f"A: {result.text}")
            print(f"   [{result.tokens_per_sec:.0f} tok/s]")

        # Final metrics
        metrics = engine.get_metrics()
        print(f"\nPipeline complete: {metrics['requests']} requests, "
              f"{metrics['total_tokens']} tokens")
```

## Recipe 10: OpenAI-Compatible Client Usage

Use the client with the familiar OpenAI-style interface:

```python
from llamatelemetry.api import LlamaCppClient

client = LlamaCppClient(base_url="http://127.0.0.1:8080")

# Multi-turn conversation
messages = [
    {"role": "system", "content": "You are a CUDA programming expert."},
    {"role": "user", "content": "What is a CUDA kernel?"},
]

# First turn
response = client.chat.completions.create(
    messages=messages,
    max_tokens=128,
    temperature=0.7,
)
assistant_msg = response.choices[0].message.content
print(f"Assistant: {assistant_msg}\n")

# Add assistant response and follow up
messages.append({"role": "assistant", "content": assistant_msg})
messages.append({"role": "user", "content": "How do I optimize it for Tesla T4?"})

response = client.chat.completions.create(
    messages=messages,
    max_tokens=128,
    temperature=0.7,
)
print(f"Assistant: {response.choices[0].message.content}")

# Check token usage
print(f"\nUsage: {response.usage.prompt_tokens} prompt + "
      f"{response.usage.completion_tokens} completion tokens")
```

## Related

- [Inference Engine](inference-engine.md) -- core engine details
- [API Client](api-client.md) -- client API reference
- [Telemetry and Observability](telemetry-observability.md) -- observability setup
- [Kaggle Environment](kaggle-environment.md) -- Kaggle-specific workflows
- [Graphistry and RAPIDS](graphistry-rapids.md) -- visualization recipes
