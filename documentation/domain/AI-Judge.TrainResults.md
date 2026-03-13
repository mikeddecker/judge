# TrainResults & Experiment metadata

Purpose
- Record model training runs, associated recipes, and validation results.

Fields & semantics
- `TrainResult` stores `step`, `recipeCode`, `recipe` (JSON), `bestEpoch`, `revalidationResults`, `lastRevalidationTime`, flags for `isBestOfAll`, `isBestOfRecipe`, `isBestOfArchitecture`, `isTestrun`, `trainEnd`, and a relation `epochs` to `TrainResultEpoch`.
- `TrainResultEpoch` stores per-epoch validation JSON in `validationResults`.

Usage
- Training jobs should create `TrainResult` entries with metadata and append `TrainResultEpoch` rows as epochs finish.
- UI shows confusion matrices, loss/metrics across epochs and highlights best epochs for each recipe.

Repro advice
- Store recipe hash, dataset snapshot, and model seed in `recipe` JSON. Keep weights and artifacts in `ENVS.DIRS.WEIGHTS` with the same hash to allow reproduction.

