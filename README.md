# Species Classifier

Fine-grained plant species classification on the iNaturalist 2021 dataset, with a VLM-rescue layer (Claude Sonnet 4.6) that escalates low-confidence ResNet predictions.

## Setup

```bash
pip install -r requirements.txt
```

Put `ANTHROPIC_API_KEY=...` in `.env` for VLM experiments.

## Entry points

- `image_classification_demo.py` — interactive demo (ResNet-101 + optional VLM rescue).
- `models/model_evaluators/` — training/evaluation scripts per architecture (`resnet_50_scratch_trained.py`, `resnet_101_scratch_trained.py`, `resnet_50_transfer.py`, `vlm_rescue_experiment.py`).
- `scripts/download_data.py` — fetch iNaturalist data.

## Layout

```
config/              constants, device selection
data/                dataset annotations + state boundary files
models/
  model_definitions/   architecture defs (ResNet-50/101, transfer)
  model_evaluators/    train + eval scripts (one per experiment)
  model_utils/         shared train/eval helpers
  trained_models/      checkpoint dirs
utils/               data loading, dataset, evaluation utilities
scripts/             data download + dataset-stats graphs
graphs_and_stats/    saved plots per experiment
experiment_results/  per-run logs and metrics
```
