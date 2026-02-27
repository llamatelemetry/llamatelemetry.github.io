# Kaggle API Reference

## Module: `llamatelemetry.kaggle`

## Main class

- `KaggleEnvironment`

Key methods:

- `KaggleEnvironment.setup(...)`
- `create_engine(model_name_or_path, ...)`
- `rapids_context()`
- `llm_context()`
- `download_model(repo_id, filename, local_dir=None)`
- `get_model_download_path()`

## Preset APIs

- `ServerPreset`
- `TensorSplitMode`
- `PresetConfig`
- `get_preset_config(...)`
- `PRESET_CONFIGS`

## Secrets APIs

- `KaggleSecrets`
- `auto_load_secrets(...)`
- `setup_huggingface_auth()`
- `setup_graphistry_auth()`

## GPU context APIs

- `GPUContext`
- `rapids_gpu(...)`
- `llm_gpu(...)`
- `single_gpu(...)`
- `get_current_gpu_context()`
- `set_gpu_for_rapids(...)`
- `reset_gpu_context(...)`

## Convenience function

- `quick_setup(**kwargs)` -> alias for `KaggleEnvironment.setup`
