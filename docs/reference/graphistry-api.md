# Graphistry API Reference

## Module: `llamatelemetry.graphistry`

## Workload and graph creation

- `GraphWorkload`
- `SplitGPUManager`
- `create_graph_from_llm_output(...)`
- `visualize_knowledge_graph(...)`

## RAPIDS helper APIs

- `RAPIDSBackend`
- `create_cudf_dataframe(...)`
- `run_cugraph_algorithm(...)`
- `check_rapids_available()`

## Connector APIs

- `GraphistryConnector`
- `register_graphistry(...)`
- `plot_graph(...)`

## Visualization APIs

- `GraphistryViz`
- `TraceVisualization`
- `MetricsVisualization`
- `create_graph_viz(...)`

## Typical call chain

1. Register Graphistry credentials.
2. Build entities/edges from LLM output.
3. Convert to graph frame objects (optionally RAPIDS-backed).
4. Plot and iterate.
