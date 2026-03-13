# Tag & TagGroup

Purpose
- Tags label videos for filtering, training splits, and result reporting.

Fields & behavior
- `Tag` stores `name` and `keywords` (comma-separated string in DB, split into a list in domain model).
- `TagGroup` groups tags for easier selection in the UI.
- Tags are linked to Videos many-to-many through `video_tag`.

Front-end
- Components: `TagConfig.vue`, `TagGroup` listing and selection in `VideoInfoContainer.vue`.
- Use tag keywords for quick search and auto-suggestion.

Notes
- Mapping: `MapToDomain.map_tag()` converts DB keywords safely to a string (avoids None) and creates `Tag` domain objects.
- Keep `keywords` normalized (lowercase, stripped) to avoid mismatches in filtering.

