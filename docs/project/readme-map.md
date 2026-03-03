# README Map

This page cross-references the main repository README sections with the corresponding deeper documentation in this site.

---

## Overview

The `llamatelemetry` README in the GitHub repository provides a concise introduction and quickstart. This documentation site expands every section into comprehensive guides, API references, and tutorials.

---

## README Section → Documentation Mapping

### Project Description

> "A CUDA-first OpenTelemetry SDK for LLM inference observability"

**Docs:**
- [Get Started: Overview](../get-started/index.md) — what llamatelemetry is, key capabilities, target audience
- [Project Architecture](architecture.md) — full 10-module architecture breakdown
- [Project File Map](file-map.md) — every file in the repository mapped to its purpose

---

### Features List

| README Feature | Detailed Documentation |
|---------------|----------------------|
| High-level `InferenceEngine` API | [Core API Reference](../reference/core-api.md) |
| Auto-download CUDA binary | [Bootstrap internals](file-map.md#_internal) |
| llama.cpp server management | [Server and Models API](../reference/server-models.md) |
| OpenAI-compatible client | [Client API Reference](../reference/client-api.md) |
| Multi-GPU split inference | [Multi-GPU and NCCL API](../reference/multigpu-nccl.md) |
| OpenTelemetry tracing + metrics | [Telemetry API Reference](../reference/telemetry-api.md) |
| 45 `gen_ai.*` semconv attributes | [Telemetry API: Semantic Conventions](../reference/telemetry-api.md#semantic-conventions-semconv) |
| Kaggle T4 x2 presets | [Kaggle API Reference](../reference/kaggle-api.md) |
| Graphistry + RAPIDS visualization | [Graphistry API Reference](../reference/graphistry-api.md) |
| GGUF quantization and conversion | [GGUF API Reference](../reference/gguf-api.md) |
| Unsloth fine-tuning integration | [Quantization and Unsloth API](../reference/quantization-unsloth.md) |
| Jupyter chat widget | [Jupyter, Chat, and Embeddings API](../reference/jupyter-chat-embeddings.md) |
| MODEL_REGISTRY (30+ models) | [Server and Models API: Registry](../reference/server-models.md#model-registry) |
| C++/CUDA extension | [CUDA and Inference API](../reference/cuda-inference-api.md) |

---

### Installation

**README shows:**

```bash
pip install llamatelemetry
```

**Docs expand to:**
- [Installation Guide](../get-started/installation.md) — pip, editable install, optional extras, CUDA requirements, binary setup
- [FAQ: Installation](faq.md#installation) — common install questions and troubleshooting

---

### Quickstart

**README shows a 5-line example:**

```python
import llamatelemetry
with llamatelemetry.InferenceEngine() as engine:
    engine.load_model("gemma-3-1b-Q4_K_M")
    result = engine.infer("Hello, world!")
    print(result.text)
```

**Docs expand to:**
- [Quickstart Guide](../get-started/quickstart.md) — step-by-step with explanations, streaming, batch inference, embeddings
- [Core API Reference](../reference/core-api.md) — full `InferenceEngine` API with all parameters

---

### Kaggle Setup

**README shows the Kaggle one-liner:**

```python
from llamatelemetry.kaggle import KaggleEnvironment
env = KaggleEnvironment()
env.quick_setup(hf_token="your-token")
```

**Docs expand to:**
- [Kaggle Quickstart](../get-started/kaggle-quickstart.md) — full Kaggle notebook walkthrough for dual-T4
- [Kaggle Environment Guide](../guides/kaggle-environment.md) — split GPU sessions, secrets, presets
- [Kaggle API Reference](../reference/kaggle-api.md) — `KaggleEnvironment`, `KaggleSecrets`, `split_gpu_session`, `ServerPreset`

---

### Multi-GPU Inference

**README shows:**

```python
from llamatelemetry.api.multigpu import kaggle_t4_dual_config
config = kaggle_t4_dual_config(model_size_b=13.0)
engine.load_model("model-Q4_K_M", multi_gpu_config=config)
```

**Docs expand to:**
- [Multi-GPU and NCCL API](../reference/multigpu-nccl.md) — `MultiGPUConfig`, `SplitMode`, `NCCLCommunicator`, all detection functions
- [Guide: CUDA Optimizations](../guides/cuda-optimizations.md) — CUDAGraph, TensorCore, FlashAttention for multi-GPU
- [Guide: Kaggle Environment](../guides/kaggle-environment.md) — split GPU session for LLM + visualization

---

### OpenTelemetry Integration

**README shows:**

```python
from llamatelemetry.telemetry import setup_telemetry
setup_telemetry(
    service_name="my-llm",
    otlp_endpoint="https://otlp.example.com/v1/traces",
)
```

**Docs expand to:**
- [Telemetry and Observability Guide](../guides/telemetry-observability.md) — end-to-end telemetry setup, metrics, exporters
- [Telemetry API Reference](../reference/telemetry-api.md) — all 45 `gen_ai.*` attributes, 5 metrics, all classes

---

### Model Management

**README shows the registry:**

```python
engine.load_model("gemma-3-4b-Q4_K_M")  # From registry
engine.load_model("/path/to/model.gguf") # Local file
engine.load_model("repo/id:filename.gguf") # HuggingFace
```

**Docs expand to:**
- [Guide: Model Management](../guides/model-management.md) — registry reference, SmartModelDownloader, VRAM planning
- [Server and Models API](../reference/server-models.md) — `MODEL_REGISTRY`, `SmartModelDownloader`, `load_model_smart`

---

### Graphistry Visualization

**README shows:**

```python
from llamatelemetry.graphistry import GraphistryConnector
connector = GraphistryConnector(server="https://hub.graphistry.com")
connector.login(username="user", password="pass")
```

**Docs expand to:**
- [Guide: Graphistry and RAPIDS](../guides/graphistry-rapids.md) — knowledge graph visualization, RAPIDS cuGraph
- [Graphistry API Reference](../reference/graphistry-api.md) — `GraphistryConnector`, graph builders, RAPIDS ops

---

### GGUF and Quantization

**README shows:**

```python
from llamatelemetry.api.gguf import quantize
quantize("model.gguf", "model-Q4_K_M.gguf", quant_type="Q4_K_M")
```

**Docs expand to:**
- [Guide: Quantization](../guides/quantization.md) — quantization strategies, choosing the right type for T4
- [GGUF API Reference](../reference/gguf-api.md) — `GGMLType`, `quantize()`, `convert_hf_to_gguf()`, `merge_lora()`
- [Quantization and Unsloth API](../reference/quantization-unsloth.md) — NF4, dynamic quant, Unsloth LoRA pipeline

---

### Unsloth Fine-Tuning

**README shows the export pipeline:**

```python
from llamatelemetry.unsloth import export_to_gguf
export_to_gguf(model, tokenizer, output_path="finetuned-Q4_K_M.gguf")
```

**Docs expand to:**
- [Guide: Unsloth Integration](../guides/unsloth.md) — full fine-tuning → GGUF → deployment pipeline
- [Quantization and Unsloth API](../reference/quantization-unsloth.md) — `UnslothLoader`, `LoRAAdapter`, `GGUFExporter`

---

### Jupyter Integration

**README shows:**

```python
from llamatelemetry.jupyter import ChatWidget
widget = ChatWidget(engine)
widget.display()
```

**Docs expand to:**
- [Guide: Jupyter Workflows](../guides/jupyter-workflows.md) — ChatWidget, streaming visualization, notebook patterns
- [Jupyter, Chat, and Embeddings API](../reference/jupyter-chat-embeddings.md) — `ChatWidget`, `ChatEngine`, `EmbeddingEngine`, `SemanticSearch`

---

### Examples

**README points to the `examples/` directory.**

**Docs expand to:**
- [Guide: Examples Cookbook](../guides/examples-cookbook.md) — annotated versions of all examples with explanations
- [Notebook Hub](../notebooks/index.md) — 18 Jupyter tutorials covering foundation to production observability

---

### Release Artifacts

**README links to GitHub releases.**

**Docs expand to:**
- [Release Artifacts](release-artifacts.md) — what is in each release archive, source vs binary distributions, how to install from a release

---

### Changelog

**README summarizes the current version.**

**Docs expand to:**
- [Changelog](changelog.md) — full annotated changelog for v0.1.0 covering all 10 modules, 18 notebooks, and the test suite

---

### Contributing

**README has a brief section.**

**Docs expand to:**
- [Contributing](contributing.md) — full contributing guide: dev setup, build instructions, test suite, code style, PR process, release process

---

### License

**MIT License** — [github.com/llamatelemetry/llamatelemetry/blob/main/LICENSE](https://github.com/llamatelemetry/llamatelemetry/blob/main/LICENSE)

---

## Documentation Site Structure

For a complete view of this documentation site's pages and sections:

```
llamatelemetry.github.io/
├── Home                         → docs/index.md
├── Get Started/
│   ├── Overview                 → get-started/index.md
│   ├── Installation             → get-started/installation.md
│   ├── Quickstart               → get-started/quickstart.md
│   └── Kaggle Quickstart        → get-started/kaggle-quickstart.md
├── Guides/
│   ├── Inference Engine         → guides/inference-engine.md
│   ├── Server Management        → guides/server-management.md
│   ├── Model Management         → guides/model-management.md
│   ├── API Client               → guides/api-client.md
│   ├── Telemetry & Observability→ guides/telemetry-observability.md
│   ├── Kaggle Environment       → guides/kaggle-environment.md
│   ├── Examples Cookbook        → guides/examples-cookbook.md
│   ├── Graphistry & RAPIDS      → guides/graphistry-rapids.md
│   ├── Quantization             → guides/quantization.md
│   ├── Unsloth Integration      → guides/unsloth.md
│   ├── CUDA Optimizations       → guides/cuda-optimizations.md
│   ├── Jupyter Workflows        → guides/jupyter-workflows.md
│   ├── Louie Knowledge Graphs   → guides/louie-knowledge-graphs.md
│   └── Troubleshooting          → guides/troubleshooting.md
├── API Reference/
│   ├── Reference Index          → reference/index.md
│   ├── Core API                 → reference/core-api.md
│   ├── Server and Models        → reference/server-models.md
│   ├── Client API               → reference/client-api.md
│   ├── GGUF API                 → reference/gguf-api.md
│   ├── Multi-GPU and NCCL       → reference/multigpu-nccl.md
│   ├── Telemetry API            → reference/telemetry-api.md
│   ├── Kaggle API               → reference/kaggle-api.md
│   ├── Graphistry API           → reference/graphistry-api.md
│   ├── Quantization & Unsloth   → reference/quantization-unsloth.md
│   ├── CUDA & Inference         → reference/cuda-inference-api.md
│   ├── Jupyter, Chat, Embeddings→ reference/jupyter-chat-embeddings.md
│   └── Louie API                → reference/louie-api.md
├── Notebooks/
│   ├── Notebook Hub             → notebooks/index.md
│   ├── Foundation Track         → notebooks/foundation.md
│   ├── Integration Track        → notebooks/integration.md
│   ├── Advanced Track           → notebooks/advanced.md
│   └── Observability Track      → notebooks/observability.md
└── Project/
    ├── Architecture             → project/architecture.md
    ├── File Map                 → project/file-map.md
    ├── Release Artifacts        → project/release-artifacts.md
    ├── FAQ                      → project/faq.md
    ├── README Map               → project/readme-map.md (this page)
    ├── Changelog                → project/changelog.md
    └── Contributing             → project/contributing.md
```
