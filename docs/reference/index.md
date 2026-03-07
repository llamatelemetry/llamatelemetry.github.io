# API Reference

Complete API reference for the `llamatelemetry` Python SDK v0.1.1. This section documents
every public class, function, dataclass, and enum across all 10 modules.

## Module Map

| Module | Description | Reference Page |
|--------|-------------|----------------|
| `llamatelemetry` | High-level `InferenceEngine`, `InferResult`, and top-level helpers | [Core API](core-api.md) |
| `llamatelemetry.server` | `ServerManager`, `ModelInfo`, `ModelManager`, `SmartModelDownloader` | [Server & Models](server-models.md) |
| `llamatelemetry.api.client` | `LlamaCppClient`, sub-APIs, `ChatEngine`, `ConversationManager`, `EmbeddingEngine`, `SemanticSearch` | [Client API](client-api.md) |
| `llamatelemetry.api.gguf` | GGUF parsing, validation, quantization matrix, `GGUFReader`, enums | [GGUF API](gguf-api.md) |
| `llamatelemetry.api.multigpu` | `MultiGPUConfig`, `SplitMode`, GPU detection and presets | [Multi-GPU & NCCL](multigpu-nccl.md) |
| `llamatelemetry.api.nccl` | `NCCLConfig`, `NCCLCommunicator`, collective operations | [Multi-GPU & NCCL](multigpu-nccl.md) |
| `llamatelemetry.telemetry` | OpenTelemetry setup, `InstrumentedLlamaCppClient`, `PerformanceMonitor`, `GpuMetricsCollector`, `GraphistryTraceExporter` | [Telemetry API](telemetry-api.md) |
| `llamatelemetry.kaggle` | `KaggleEnvironment`, `ServerPreset`, pipeline config, secrets | [Kaggle API](kaggle-api.md) |
| `llamatelemetry.graphistry` | `GraphistrySession`, `GraphistryBuilders`, `GraphistryViz`, `RAPIDSBackend`, `SplitGPUManager` | [Graphistry API](graphistry-api.md) |
| `llamatelemetry.quantization` | `NF4Quantizer`, `GGUFConverter`, `DynamicQuantizer`, strategies | [Quantization & Unsloth](quantization-unsloth.md) |
| `llamatelemetry.unsloth` | `UnslothModelLoader`, `UnslothExporter`, `LoRAAdapter` | [Quantization & Unsloth](quantization-unsloth.md) |
| `llamatelemetry.cuda` | `CUDAGraph`, `TritonKernel`, `TensorCoreConfig` | [CUDA & Inference](cuda-inference-api.md) |
| `llamatelemetry.inference` | `FlashAttentionConfig`, `KVCache`, `ContinuousBatching`, `BatchInferenceOptimizer` | [CUDA & Inference](cuda-inference-api.md) |
| `llamatelemetry.jupyter` | Jupyter widgets, streaming helpers, visualization | [Jupyter, Chat & Embeddings](jupyter-chat-embeddings.md) |
| `llamatelemetry.chat` | `ChatEngine`, `ConversationManager`, `Message` | [Jupyter, Chat & Embeddings](jupyter-chat-embeddings.md) |
| `llamatelemetry.embeddings` | `EmbeddingEngine`, `SemanticSearch`, `TextClustering`, similarity functions | [Jupyter, Chat & Embeddings](jupyter-chat-embeddings.md) |
| `llamatelemetry.louie` | `LouieClient`, `KnowledgeExtractor`, `KnowledgeGraph`, entity/relation enums | [Louie API](louie-api.md) |

## Quick Links by Category

### Core

- [InferenceEngine](core-api.md#inferenceengine) -- primary entry point for all inference
- [InferResult](core-api.md#inferresult) -- result wrapper returned by all inference methods
- [ServerManager](server-models.md#servermanager) -- llama-server process lifecycle
- [LlamaCppClient](client-api.md#llamacppclient) -- low-level HTTP client for llama-server

### Model Management

- [ModelInfo](server-models.md#modelinfo) -- GGUF metadata parser
- [ModelManager](server-models.md#modelmanager) -- scan and filter local models
- [SmartModelDownloader](server-models.md#smartmodeldownloader) -- VRAM-aware download
- [GGUFReader](gguf-api.md#ggufreader) -- memory-mapped GGUF file reader

### Multi-GPU

- [MultiGPUConfig](multigpu-nccl.md#multigpuconfig) -- GPU split configuration
- [NCCLCommunicator](multigpu-nccl.md#ncclcommunicator) -- NCCL collective operations

### Observability

- [setup_grafana_otlp](telemetry-api.md#setup_grafana_otlp) -- one-call telemetry setup
- [PerformanceMonitor](telemetry-api.md#performancemonitor) -- real-time performance tracking
- [GpuMetricsCollector](telemetry-api.md#gpumetricscollector) -- GPU metric collection
- [GraphistryTraceExporter](telemetry-api.md#graphistrytraceexporter) -- trace visualization

### Environments

- [KaggleEnvironment](kaggle-api.md#kaggleenvironment) -- Kaggle platform detection
- [ServerPreset](kaggle-api.md#serverpreset-enum) -- environment-specific presets

### Graph & Knowledge

- [GraphistrySession](graphistry-api.md#graphistrysession) -- Graphistry connection
- [GraphistryBuilders](graphistry-api.md#graphistrybuilders) -- graph construction helpers
- [LouieClient](louie-api.md#louieclient) -- natural language graph queries
- [KnowledgeExtractor](louie-api.md#knowledgeextractor) -- entity/relationship extraction

### Quantization & Fine-tuning

- [NF4Quantizer](quantization-unsloth.md#nf4quantizer) -- NormalFloat4 quantization
- [GGUFConverter](quantization-unsloth.md#ggufconverter) -- model-to-GGUF conversion
- [DynamicQuantizer](quantization-unsloth.md#dynamicquantizer) -- VRAM-aware quantization
- [UnslothModelLoader](quantization-unsloth.md#unslothmodelloader) -- Unsloth model loading
- [LoRAAdapter](quantization-unsloth.md#loraadapter) -- LoRA adapter management

### CUDA & Inference

- [CUDAGraph](cuda-inference-api.md#cudagraph) -- CUDA graph capture and replay
- [FlashAttentionConfig](cuda-inference-api.md#flashattentionconfig) -- flash attention setup
- [KVCache](cuda-inference-api.md#kvcache) -- KV cache management
- [ContinuousBatching](cuda-inference-api.md#continuousbatching) -- continuous batching

### Interactive

- [ChatEngine](jupyter-chat-embeddings.md#chatengine) -- multi-turn conversation
- [EmbeddingEngine](jupyter-chat-embeddings.md#embeddingengine) -- text embeddings
- [SemanticSearch](jupyter-chat-embeddings.md#semanticsearch) -- vector similarity search

## Conventions

- All classes support Python context managers where noted with `__enter__` / `__exit__`.
- Optional dependencies degrade gracefully; import errors are caught and features are disabled.
- GPU operations require CUDA 12.x and a compute capability >= 7.5 device (e.g., Tesla T4).
- All timeout parameters are in seconds unless otherwise noted.
- All `**kwargs` are forwarded to the underlying llama-server HTTP API.

## Version

This reference documents `llamatelemetry` **v0.1.1** (released 2026-02-02).
