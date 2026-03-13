# Frames & FrameLabels

Concepts
- Frame: a single video frame indexed by `frameNr`.
- FrameLabel: a labeled bounding box placed on a frame with metadata (x, y, width, height, label type).

Label types
- `FrameLabelType` table stores types (e.g. foreground-person, background-person). It is initialized by `VideoService.initiate()` when empty.

Repository behaviour
- `add_frameInfo` enforces `ValueHelper` validation and persists a `FrameLabel`.
- `remove_frameInfo` finds the closest matching label (by Euclidean distance) and deletes it.
- `get_team_boxes` computes aggregated boxes per video/frame using SQL aggregation (returns bounding box extents and centers).

UI / UX recommendations
- Provide an editing mode where labels can be moved/adjusted; calculate nearest-match for deletes to avoid accidental removal.
- Expose label types in the UI using `GET /frameLabelTypes`.

Integration
- Frame labels are the basis for localization training and team-box aggregation used by downstream skill detection.

