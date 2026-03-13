# Jobs (training & prediction)

Jobs represent automated or scheduled work performed by the ML stack: training, prediction, full pipelines.

DB model
- Table: `Jobs` with fields: `id`, `type` (TRAIN/PREDICT), `step` (LOCALIZE/SEGMENT/SKILL/FULL), `job_arguments` (JSON), `status`, `status_details`, timestamps.

Lifecycle
- Typical flow: create job -> JobService validates/records -> worker (or local process) executes -> updates `status` and `status_details`.
- Status values are free-form strings in code; recommend standardized statuses: `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCELLED`.

Job types & steps
- `JOB_TYPES = ['TRAIN','PREDICT']` and `JOB_STEPS = ['LOCALIZE','SEGMENT','SKILL','FULL']` (see `api/config.py`).

Integration with computervision
- `computervision/Trainer.py` and `Predictor.py` contain the real training/predict logic.
- Jobs may write artifacts to `ENVS.DIRS.GENERATED_VIDEODATA` and weights to `ENVS.DIRS.WEIGHTS`.
- Predictions are saved per-video as: `{videoId}_skills_<model>.json` and `{videoId}_raw_boxes.json`.

Operational notes
- Jobs can be launched via API; in production consider using a queue (Redis/RabbitMQ) + worker pool for resiliency.
- Keep job logs and artifacts (model weights, validation metrics) under a versioned directory.



