# Implementing Permissions

## Status: Implemented

The permissions system has been implemented. See `documentation/features/AI-Judge.Permissions.md` for the full design.

## What was added

- **`api/repository/models.py`** — Four new ORM models: `AccountCapability`, `GroupMembership`, `AccessGrant`, `AccountBlock`. The `Account` model gains two new fields: `accountType` and `owner_id`. Groups are `Account` rows with `accountType='group'` — no separate table needed.
- **`api/repository/permissionRepo.py`** — Data-access layer (CRUD + 7-step access-check helper).
- **`api/services/permissionService.py`** — Business-logic layer (access resolution, group/grant/block management).
- **`api/routers/permissionRouter.py`** — REST endpoints (see below).
- **`api/tests/test_permissions.py`** — Unit tests for the service layer.
- **`web/src/services/permissionService.js`** — Frontend API service.
- **`web/src/views/PermissionsView.vue`** — Vue3 permissions management page (groups, grants, blocks).

## Account types

`Account.accountType` can be:

| Value | Description |
|---|---|
| `user` | Regular user (default) |
| `group` | Named group — members managed via `GroupMembership` |
| `team` | Sports team / club |
| `organisation` | Organisation/federation |
| `admin` | Administrator |

Group accounts use `Account.owner_id` to record who created them. They cannot log in directly (synthetic email + unusable password hash).

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/capabilities/<account_id>` | Get capability row for an account |
| PUT | `/capabilities/<account_id>` | Upsert capabilities (admin operation) |
| GET | `/groups` | List group accounts owned by caller |
| POST | `/groups` | Create a group account |
| DELETE | `/groups/<group_id>` | Delete a group account |
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

Auto-generate and apply:

```bash
flask db migrate -m "add permissions and account type"
flask db upgrade
```

## Security notes

- Capability rows should only be created/updated by admins. The `PUT /capabilities/<account_id>` endpoint sets `granted_by` to the calling session account — restrict this endpoint at the infrastructure level (IP whitelist or admin role) until an admin concept is added to the Account model.
- Blocks are always enforced before grants.
