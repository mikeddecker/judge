# Folder (domain)

## Purpose
- Organize videos in a hierarchical tree. Folders can be nested and are used to group videos for browsing, training splits, and permission scoping.

## Fields & constraints
- `id` — binary UUID stored in DB (helper `UUIDType` in `repository/models.py`).
- `name` — folder name, unique per parent folder enforced by the DB unique constraint `_name_parent_unique_constraint`.
- `parentId` — nullable, references another `Folders.id` for nesting.
- `is_train` — boolean flag (default `True`) indicating whether videos in the folder are considered training material; used by import/ML splits and UI filters.

## Service behaviour & invariants
- Creation: `FolderService` and `folderRepo` validate parent existence and unique-name-per-parent constraints.
- Deletion: `FolderService` prevents deleting folders that contain subfolders or videos; it raises `PermissionError` when invariants are violated (see `api/services/folderService.py`).
- Path resolution: `VideoInfo.get_relative_video_path()` and repo code use `Folder.get_relative_path()` to resolve video file locations on disk under `ENVS.DIRS.VIDEOS`.

## Permissions and privacy
- Videos are private by default; folder-level grants are supported conceptually but video-level grants prevail when present (see `documentation/features/AI-Judge.Permissions.md`).
- Ownership: folders belong to owners (accounts/teams) in higher-level designs; any permission checks should be enforced in the service layer (not in repository).

Front-end behaviour
- Folder tree navigation and creation are provided by `FolderContainer.vue` and `FolderInfo.vue`.
- The UI should only enable create/delete operations for users with appropriate capabilities (`can_manage_members` / `can_edit_video` depending on your policy).

Developer notes
- Repo-level: `api/repository/models.py` defines the `Folders` table and unique constraint — keep migrations in sync when altering schema.
- Tests: add unit tests for folder creation, renaming (if added), and safe deletion (attempt delete when subfolders/videos exist should raise `PermissionError`).

Recommendations
- Enforce folder ownership: store `owner_id` on `Folders` when multi-tenant behavior is needed and use service-layer checks to validate operations against `session['account_id']`.
- Inherit/override access: choose a clear precedence (video-specific > folder-specific > global). Document and test resolution logic.
- Add audit logging for folder creation/deletion to help trace accidental removals.

