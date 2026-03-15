# AI-Judge ComputerVision: Predict and Train Method Signatures

**Date:** March 14, 2026  
**Version:** 2.0 (Post-UUID Migration)

## Overview

This document describes the current API signatures for the Predictor and Trainer classes in the computervision module, along with the expected job argument formats from the API.

## Predictor Class

### Method: `predict()`

**Signature:**
```python
def predict(self, type: str, videoId: UUID, recipename: str,
            modelparams: dict = None, saveAsVideo: bool = False,
            date: str = None, weights: str = 'best') -> None:
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| type | str | Yes | Prediction type: `'LOCALIZE'`, `'SEGMENT'`, `'SKILL'`, `'FULL'`, or `'SEGMENT_SKILL'` |
| videoId | UUID | Yes | UUID object of the video to predict on (passed from job_arguments) |
| recipename | str | Yes | Recipe/model name to use (e.g., `'MViT_extra_dense'`) |
| modelparams | dict | No | Model parameters from RECIPES dict containing timesteps, batch_size, balancedType |
| saveAsVideo | bool | No | Whether to save predictions as annotated video file (default False) |
| date | str | No | Date string for tracking predictions |
| weights | str | No | Which model weights to load: `'best'` or path to custom weights (default 'best') |

**Prediction Types:**

- **LOCALIZE**: Run YOLO model to detect jumper locations in video
- **SEGMENT**: Predict action segments (skill boundaries) using PyTorch model
- **SKILL**: Predict individual skill attributes (type, level, turner involvement, etc.)
- **FULL**: Run complete pipeline: LOCALIZE → SEGMENT → SKILL
- **SEGMENT_SKILL**: Run SEGMENT and SKILL, combining results

**Job Format:** `{"type": "PREDICT", "step": "SKILL", "videoId": "<UUID-string>", "model": "<recipe>"}`

### Internal Methods

- `__predict_location()` - YOLO-based localization, saves bounding boxes
- `__predict_segments_pytorch()` - Segment boundaries via PyTorch model
- `__predict_skills_pytorch()` - Skill attributes via PyTorch model

---

## Trainer Class

**Method:** `train(step: str, recipename: str, speedmode: str = 'fast', job_arguments: dict = {})`

**Steps:** LOCALIZE (YOLO), SEGMENT (PyTorch), SKILL (PyTorch)
**Speedmode:** 'fast' or 'slow' (affects hyperparameters)

**Job Format:** `{"type": "TRAIN", "step": "SKILL", "recipe": "<recipe>", "speedmode": "slow"}`

## Data Flow

**Prediction:** Job (videoId string) → JobExecuter (convert to UUID) → Predictor → RepoGeneral queries

**Training:** Job arguments (recipe name) → JobExecuter → Trainer → Training pipeline

## Key Details

- **videoId:** Must be UUID object, not string
- **Model parameters:** Includes timesteps, batch_size, balancedType
- **Database queries:** All use parameterized syntax with params dict

## Troubleshooting

- **UUID attribute errors:** Ensure videoId is UUID object, not string or binary
- **SQL errors:** Verify parameterized syntax (`:paramname`) and params dict
- **VideoNames KeyError:** Ensure index uses UUID objects
- **Incorrect predictions:** Verify modelparams match RECIPES definition

