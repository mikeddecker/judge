# Machine Learning Models & artifacts

Overview
- The repository contains model training, prediction, and recipe configuration in `computervision/`.
- Model weights and generated predictions are stored under `ENVS.DIRS.GENERATED`.

Key files / folders
- `computervision/Trainer.py` — training orchestration.
- `computervision/Predictor.py` — inference wrapper used to create per-video prediction JSON.
- `computervision/weights/` — local weights storage (configured by env vars).
- `machine_learning_recipes.json` and `recipes.json` define recipes and parameters used by training jobs.

Prediction outputs
- Skills predictions: saved as `{videoId}_skills_<model>.json`.
- Localization boxes: saved as `{videoId}_raw_boxes.json`.

Models referenced
- Default best-model name used by API: `MViT` (see `videoService.getVideoPredictions`).
- The codebase is model-agnostic; add new model names and training recipes to `recipes.json` and `RECIPES` in `config.py`.

Recommendations
- Version-weighting: store weights with a timestamp and recipe hash to reproduce experiments.
- Metadata: store training metadata (dataset snapshot, seed, hyperparameters) in `TrainResults` (table present in `repository/models.py`).


