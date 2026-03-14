# Implementing Permissions

## Status: Implemented

The permissions system has been implemented. See `documentation/features/AI-Judge.Permissions.md` for the full design.

## What was added

- **`api/repository/models.py`** — five new ORM models: `AccountCapability`, `Group`, `GroupMembership`, `AccessGrant`, `AccountBlock`.
- **`api/migrations/versions/c4d5e6f7a8b9_add_permissions.py`** — Alembic migration for all five tables.
- **`api/repository/permissionRepo.py`** — data-access layer (CRUD + 7-step access-check helper).
- **`api/services/permissionService.py`** — business-logic layer (access resolution, group management, grants, blocks).
- **`api/routers/permissionRouter.py`** — REST endpoints (see below).
- **`api/tests/test_permissions.py`** — unit tests for the service layer.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/capabilities/<account_id>` | Get capability row for an account |
| PUT | `/capabilities/<account_id>` | Upsert capabilities (admin operation) |
| GET | `/groups` | List groups owned by caller |
| POST | `/groups` | Create a group |
| DELETE | `/groups/<group_id>` | Delete a group |
| POST | `/groups/<group_id>/members` | Add member to group |
| DELETE | `/groups/<group_id>/members` | Remove member from group |
| GET | `/access-grants` | List caller's access grants |
| POST | `/access-grants` | Create an access grant |
| DELETE | `/access-grants/<grant_id>` | Revoke a grant |
| GET | `/blocks` | List accounts blocked by caller |
| POST | `/blocks` | Block an account |
| DELETE | `/blocks/<account_id>` | Unblock an account |

## Access resolution order

1. Content is public (`is_public = True`) → **ALLOW**
2. Requester is blocked by owner → **DENY**
3. Requester is the owner → **ALLOW**
4. A valid `AccessGrant` exists (not expired, covers the content) → **ALLOW**
5. Default → **DENY**

## Running migrations

```bash
make migrate   # or: docker compose exec api flask db upgrade
```

## Security notes

- Capability rows should only be created/updated by admins. The `PUT /capabilities/<account_id>` endpoint sets `granted_by` to the calling session account — restrict this endpoint at the infrastructure level (IP whitelist or admin role) until an admin concept is added to the Account model.
- Blocks are always enforced before grants.

