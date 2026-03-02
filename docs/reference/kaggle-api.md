# Kaggle API

`llamatelemetry.kaggle` provides Kaggle-specific utilities and presets.

## Environment detection

- `is_kaggle()` — detect Kaggle runtime
- `get_kaggle_environment()` — return environment metadata

## Presets and configuration

- `kaggle_t4_dual_config()` — recommended split for dual T4
- `colab_t4_single_config()` — single T4 defaults

## Secrets

- `KaggleSecrets` — read secrets from Kaggle UI

## Pipeline helpers

- `KagglePipeline` — one-stop environment preparation

## Related docs

- [Kaggle Quickstart](../get-started/kaggle-quickstart.md)
- [Kaggle Environment Guide](../guides/kaggle-environment.md)
