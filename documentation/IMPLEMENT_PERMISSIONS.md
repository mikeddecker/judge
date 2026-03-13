```markdown
# Implementing Permissions — scaffold and migration plan

Goal
- Provide a concrete starting point (SQLAlchemy models + migration plan) for `AccountCapability`, `AccessGrant`, and `AccountBlock` as described in `documentation/features/AI-Judge.Permissions.md`.

Suggested SQLAlchemy models (place in `api/repository/models.py` or a new `permissions.py` and import in `models.py`):

```python
from sqlalchemy import Column, Boolean, Integer, Text, DateTime, Enum, UniqueConstraint, CheckConstraint, ForeignKey
from sqlalchemy.dialects.mysql import JSON
from repository.db import db
from repository.models import UUIDType
import uuid

class AccountCapability(db.Model):
    __tablename__ = 'AccountCapability'
    id = db.Column(UUIDType, primary_key=True, default=lambda: uuid.uuid4().bytes)
    account_id = db.Column(UUIDType, db.ForeignKey('Accounts.id'), nullable=False, unique=True)

    can_upload_video = Column(Boolean, nullable=False, default=False)
    can_edit_video = Column(Boolean, nullable=False, default=False)
    can_label_video = Column(Boolean, nullable=False, default=False)
    can_train_model = Column(Boolean, nullable=False, default=False)
    # limits
    max_video_uploads = Column(Integer, nullable=False, default=100)
    extra_flags = Column(JSON, nullable=True)

    granted_by = Column(UUIDType, db.ForeignKey('Accounts.id'), nullable=False)
    granted_at = Column(DateTime, nullable=False)
    granted_until = Column(DateTime, nullable=True)
    granted_reason = Column(Text, nullable=True)

class AccessGrant(db.Model):
    __tablename__ = 'AccessGrant'
    id = db.Column(UUIDType, primary_key=True, default=lambda: uuid.uuid4().bytes)
    owner_id = db.Column(UUIDType, db.ForeignKey('Accounts.id'), nullable=False)

    granted_to = Column(Enum('everyone','account','group', name='grantedto'), nullable=False)
    target_account_id = Column(UUIDType, db.ForeignKey('Accounts.id'), nullable=True)
    target_group_id = Column(UUIDType, db.ForeignKey('TagGroups.id'), nullable=True) # All

    video_id = Column(UUIDType, db.ForeignKey('Videos.id'), nullable=True)
    folder_id = Column(UUIDType, db.ForeignKey('Folders.id'), nullable=True)

    can_view = Column(Boolean, nullable=False, default=False)
    can_label = Column(Boolean, nullable=False, default=False)
    can_download = Column(Boolean, nullable=False, default=False)

    granted_by = Column(UUIDType, db.ForeignKey('Accounts.id'), nullable=False)
    granted_at = Column(DateTime, nullable=False)
    granted_until = Column(DateTime, nullable=True)

    __table_args__ = (
        CheckConstraint("NOT (video_id IS NOT NULL AND folder_id IS NOT NULL)", name='ck_access_grant_scope'),
    )

class AccountBlock(db.Model):
    __tablename__ = 'AccountBlock'
    id = db.Column(UUIDType, primary_key=True, default=lambda: uuid.uuid4().bytes)
    blocker_id = Column(UUIDType, db.ForeignKey('Accounts.id'), nullable=False)
    blocked_id = Column(UUIDType, db.ForeignKey('Accounts.id'), nullable=False)
    blocked_at = Column(DateTime, nullable=False)

    __table_args__ = (
        UniqueConstraint('blocker_id', 'blocked_id'),
        CheckConstraint('blocker_id != blocked_id', name='ck_block_no_self'),
    )
```

Migration plan (Alembic)
1. Create a migration file `versions/xxxx_add_permissions_tables.py` that creates the three tables above using raw SQL or SQLAlchemy `op.create_table()`.
2. Run `flask db upgrade` in your environment to apply.

Wiring into code
- `repository/accountRepo.py`: add a helper `get_capabilities(account_id)` returning `AccountCapability` row.
- `services/videoService.py` `__authorize_update`: replace current hardblock for `training` with a capability check:

```python
cap = AccountRepo.get_capabilities(account.id)
if any(k in user_data for k in training_keys) and not cap.can_train_model:
    raise PermissionError('Only accounts with can_train_model can change training flag')
```

Testing
- Add unit tests for capability checks in `api/tests/test_service_video.py`.

Security
- Ensure migration and model code do not leak capability flags in public API responses. Only admin views should return raw capability rows.

```

