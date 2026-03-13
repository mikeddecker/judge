Core permission flags — stored flat for fast querying
# Admin/application grants these based on subscription. 
~20 booleans is fine as columns.

# Base permissions

- granted_by (FK_account)
- granted_at (Datetime)
- granted_until: (Datetime, nullable)
- granted_reason: (Text, nullable)

- target_account_id (can not be both null)
- target_group_id (can not be both null)
- target_type (owner, representative, group, friend, follower, individual)

Ideally if target_type is group, then target_group_id must be filled in.
#TODO Fit in how to say: organisation/team/club representatives can see videos.
#TODO Add block certain people/groups. Block always wins.
# Granted by admin / subscription
- can_upload_video
- can_edit_video
- can_label_video
- can_see_video_actions (💲)
- can_see_tags
- can_edit_tags
- can_see_video_tags
- can_see_video_labels
- can_train_model
- can_manage_members (teams/orgs/groups)
- can_manage_representatives (teams/orgs/groups)
- can_export_model
- can_invite_users

# Limits (from subscription)
- max_video_uploads (INT, default=100)
- max_video_size_mb (INT, default=2048)
- max_video_duration_seconds (SMALLINT, default=120)
- max_storage_gb (default=20)

# Extended/future permissions
e.g. {"can_use_beta_features": true, "api_access": true}

# Additional info about permissions

## can_edit_video

Edit name, folders & tags

## can_manage_members
Wil probably not be given to individual users. Doesn't make any sense.

## can_manage_representatives
Wil probably not be given to individual users. Doesn't make any sense.

## Notes — implementation guidance

- Prefer a single table `AccountCapability` (one row per account) to store the boolean capability flags
	(e.g. `can_edit_video`, `can_train_model`) and limits (max uploads, max size, ...).
- Videos are private by default. Access is granted explicitly by owner/representative via
	an `AccessGrant` (or `AccessGrant`-like table) where grants may target a specific `video_id` or a `folder_id`.
- Video-level grants prevail over folder-level grants. If a video has an explicit grant, it overrides
	the folder's access rules for that video.
- Block records (e.g. `AccountBlock`) always win over grants.

## Minimal runtime authorization rules (current code)

- Router layer: authenticates user identity and provides `account_id` via Flask `session` (see `/auth/login`).
- Service layer: enforces authorization checks. The `VideoService` should verify:
	- caller is authenticated
	- requested fields are allowed to be updated (field-level mapping)
	- sensitive flags such as `training` require elevated privileges (admins or subscription tier)

If you want me to implement DB models for `AccountCapability`, `AccessGrant` and `AccountBlock`,
I can scaffold SQLAlchemy models and example migration code.

# Code examples
Enum or str?

```python
class AccountType(str): 
	ADMIN = 'admin' 
	USER = 'user' 
	TEAM = 'team' 
	ORGANISATION = 'organisation'
	
...

class RelationshipType(str): 
	FRIEND = 'friend' 
	MEMBER = 'member' 
	REPRESENTATIVE = 'representative' 
	FOLLOWER = 'follower'

__table_args__ = ( CheckConstraint("account_type IN ('admin','user','team','organisation')"), )
```

