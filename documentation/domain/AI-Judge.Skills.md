# Skills (domain)

Skills are labeled segments inside a video representing a judged element (e.g. jump, wrap, transition).

DB model
- Table: `Skills` (columns: `id`, `videoId`, `frameStart`, `frameEnd`, `skillinfo` JSON).

Business rules enforced in code
- Frame numbers validated with `ValueHelper`.
- Minimum length: the service requires `frameEnd - frameStart > 4`.
- No overlap: `VideoService` checks `videoinfo.has_skill_overlap(...)` before insertion.
- `skillinfo` is stored as JSON and should follow the project's `LayerComposition` schema (see `api/helpers/ConfigHelper.py` and `repository/models.py` for layer definitions).

Training & completed flags
- `Videos.completed_skill_labels` tracks whether skill labels are finalized for a video.
- `Videos.is_train` (and `is_train`) indicate whether a video is used for training. Changing this flag is gated by authorization (service-level checks).

Recommended front-end behaviour
- Provide a UI for selecting start/end frames and a structured form for `skillinfo` (guided by layer composition).
- Show warnings when creating skills overlapping existing ones.

Integration with ML
- Skill labels are used as targets for model training and for evaluation metrics.
- During training, the ML stack reads `Skills` JSON via export scripts or direct DB access.

