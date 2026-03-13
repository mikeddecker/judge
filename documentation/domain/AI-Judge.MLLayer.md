
# ML Layer & LayerComposition

Purpose
- Layers describe target heads used for recognition tasks (boolean, categorical, numerical). `LayerComposition` composes multiple layers into a skill-level target schema.

Key concepts
- `Layer`: name, type (`boolean`/`categorical`/`numerical`), min/max/step for numerical and allowed values for categorical via `LayerValue`.
- `LayerValue`: allowed category entries for categorical layers.
- `LayerComposition`: ties layers together for a named composition used to build `skillinfo` payloads.

Front-end
- Components: `LayerComposition.vue`, `LayerCompositionElementCard.vue`, `LayerValueSelector.vue` to build and reuse compositions when labeling skills.

Integration notes
- `ValueHelper` validates values for layers before inserting skills. Training recipes expect consistent `LayerComposition` definitions.

