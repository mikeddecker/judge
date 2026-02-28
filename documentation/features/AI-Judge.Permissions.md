2026-02-27 - based on claude sonnet 4.6
```table-of-contents
```

# Overview

The schema is split into four concerns:

1. **AccountCapability** — what an account is _allowed to do_ (upload, train, export, etc.) — granted by admin/subscription
2. **Group** + **GroupMembership** — named sets of accounts that can receive shared access
3. **AccessGrant** — who can _see or interact with_ whose content — granted by owner/admin/representative
4. **AccountBlock** — blocks, always win over grants

---

# Entity Relationship Summary

```
Account ──< AccountCapability          (1 capability row per account)
Account ──< GroupMembership >── Group  (many-to-many)
Account ──< AccessGrant                (owner grants access to others)
Account ──< AccountBlock               (owner blocks others)

AccessGrant.target_account_id ──> Account  (nullable)
AccessGrant.target_group_id   ──> Group    (nullable)
AccessGrant.video_id          ──> Video    (nullable — if NULL: applies to all owner's content)
AccessGrant.folder_id         ──> Folder   (nullable — if NULL: applies to all owner's content)
```

---

# 1. AccountCapability

> One row per account. Granted/managed by admin or subscription logic. Never by users.

```python
class AccountCapability(Base):
    __tablename__ = "account_capability"

    id              = Column(UUID, primary_key=True, default=uuid4)
    account_id      = Column(UUID, ForeignKey("account.id"), nullable=False, unique=True)

    # --- Granted by admin / subscription tier ---
    can_upload_video        = Column(Boolean, default=False, nullable=False)
    can_edit_video          = Column(Boolean, default=False, nullable=False)  # name, folder, tags
    can_label_video         = Column(Boolean, default=False, nullable=False)
    can_see_video_actions   = Column(Boolean, default=False, nullable=False)  # 💲 paid feature
    can_see_tags            = Column(Boolean, default=False, nullable=False)
    can_edit_tags           = Column(Boolean, default=False, nullable=False)
    can_see_video_tags      = Column(Boolean, default=False, nullable=False)
    can_see_video_labels    = Column(Boolean, default=False, nullable=False)
    can_train_model         = Column(Boolean, default=False, nullable=False)
    can_export_model        = Column(Boolean, default=False, nullable=False)
    can_manage_members      = Column(Boolean, default=False, nullable=False)  # teams/orgs only
    can_manage_representatives = Column(Boolean, default=False, nullable=False)  # teams/orgs only
    can_invite_users        = Column(Boolean, default=False, nullable=False)

    # --- Limits (from subscription) ---
    max_video_uploads           = Column(SmallInteger, default=100,  nullable=False)
    max_video_size_mb           = Column(Integer,      default=2048, nullable=False)  # INTEGER: up to ~2TB
    max_video_duration_seconds  = Column(SmallInteger, default=120,  nullable=False)
    max_storage_gb              = Column(SmallInteger, default=20,   nullable=False)

    # --- Extended / future flags ---
    extra_flags = Column(JSONB, nullable=True)  # {"api_access": true, "can_use_beta": true}

    # --- Grant metadata ---
    granted_by      = Column(UUID, ForeignKey("account.id"), nullable=False)
    granted_at      = Column(DateTime(timezone=True), nullable=False, default=func.now())
    granted_until   = Column(DateTime(timezone=True), nullable=True)
    granted_reason  = Column(Text, nullable=True)
```

**Notes:**

- `can_manage_members` and `can_manage_representatives` should only be granted to `team`/`organisation` account types. Enforce this in application logic or a DB trigger.
- Use `INTEGER` instead of `SMALLINT` for `max_video_size_mb` — `SMALLINT` caps at 32,767 MB (~32 GB).

---

# 2. Group + GroupMembership

> Groups are independent of account type. Any account (user, team, org) can be a member of any group.

```python
class Group(Base):
    __tablename__ = "group"

    id          = Column(UUID, primary_key=True, default=uuid4)
    owner_id    = Column(UUID, ForeignKey("account.id"), nullable=False)  # who created/owns this group
    name        = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at  = Column(DateTime(timezone=True), nullable=False, default=func.now())


class GroupMembership(Base):
    __tablename__ = "group_membership"

    id          = Column(UUID, primary_key=True, default=uuid4)
    group_id    = Column(UUID, ForeignKey("group.id"), nullable=False)
    account_id  = Column(UUID, ForeignKey("account.id"), nullable=False)
    added_at    = Column(DateTime(timezone=True), nullable=False, default=func.now())
    added_by    = Column(UUID, ForeignKey("account.id"), nullable=False)

    __table_args__ = (
        UniqueConstraint("group_id", "account_id"),  # no duplicate memberships
    )
```

**Example groups an owner might create:**

- "My coaching staff" → representatives of Team A
- "League officials" → org accounts who should see all match videos
- "Friends" → individual users with casual access

---

# 3. AccessGrant

> Controls _who can see or interact with_ the owner's content. Granted by: video owner, admin, or representatives of the owner.

```python
class GrantedTo(str, Enum):
    EVERYONE        = "everyone"       # public — no target FK needed
    ACCOUNT         = "account"        # specific account
    GROUP           = "group"          # all members of a group


class AccessGrant(Base):
    __tablename__ = "access_grant"

    id          = Column(UUID, primary_key=True, default=uuid4)
    owner_id    = Column(UUID, ForeignKey("account.id"), nullable=False)  # whose content this protects

    # --- Who is being granted access ---
    granted_to          = Column(Enum(GrantedTo), nullable=False)
    target_account_id   = Column(UUID, ForeignKey("account.id"), nullable=True)
    target_group_id     = Column(UUID, ForeignKey("group.id"), nullable=True)

    # --- What content this applies to (NULL = all owner content) ---
    # At most one of these should be set. If both NULL: applies globally to owner's content.
    video_id    = Column(UUID, ForeignKey("video.id"), nullable=True)
    folder_id   = Column(UUID, ForeignKey("folder.id"), nullable=True)

    # --- What is actually granted ---
    can_view        = Column(Boolean, default=False, nullable=False)
    can_comment     = Column(Boolean, default=False, nullable=False)
    can_label       = Column(Boolean, default=False, nullable=False)  # e.g. reps labeling match footage
    can_download    = Column(Boolean, default=False, nullable=False)

    # --- Relationship type (for display/filtering, not enforcement) ---
    relationship_type = Column(
        Enum("friend", "member", "representative", "follower", "individual", name="relationship_type"),
        nullable=True
    )

    # --- Grant metadata ---
    granted_by      = Column(UUID, ForeignKey("account.id"), nullable=False)
    granted_at      = Column(DateTime(timezone=True), nullable=False, default=func.now())
    granted_until   = Column(DateTime(timezone=True), nullable=True)
    granted_reason  = Column(Text, nullable=True)

    __table_args__ = (
        # Exactly one target must be set when granted_to != EVERYONE
        CheckConstraint("""
            (granted_to = 'everyone' AND target_account_id IS NULL AND target_group_id IS NULL)
            OR
            (granted_to = 'account' AND target_account_id IS NOT NULL AND target_group_id IS NULL)
            OR
            (granted_to = 'group'   AND target_group_id IS NOT NULL AND target_account_id IS NULL)
        """, name="ck_access_grant_target"),

        # At most one scope (video or folder), not both
        CheckConstraint(
            "NOT (video_id IS NOT NULL AND folder_id IS NOT NULL)",
            name="ck_access_grant_scope"
        ),
    )
```

---

# 4. AccountBlock

> Separate table. Block always wins over any AccessGrant.

```python
class AccountBlock(Base):
    __tablename__ = "account_block"

    id          = Column(UUID, primary_key=True, default=uuid4)
    blocker_id  = Column(UUID, ForeignKey("account.id"), nullable=False)
    blocked_id  = Column(UUID, ForeignKey("account.id"), nullable=False)
    blocked_at  = Column(DateTime(timezone=True), nullable=False, default=func.now())
    reason      = Column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("blocker_id", "blocked_id"),
        CheckConstraint("blocker_id != blocked_id", name="ck_block_no_self"),
    )
```

---

# 5. Video / Folder (public flag)

> `is_public` lives on the content itself. This is simpler and faster than a wildcard `AccessGrant` row.

```python
class Video(Base):
    __tablename__ = "video"

    id          = Column(UUID, primary_key=True, default=uuid4)
    owner_id    = Column(UUID, ForeignKey("account.id"), nullable=False)
    folder_id   = Column(UUID, ForeignKey("folder.id"), nullable=True)
    is_public   = Column(Boolean, default=False, nullable=False)  # visible to everyone, no grant needed
    # ... other fields


class Folder(Base):
    __tablename__ = "folder"

    id          = Column(UUID, primary_key=True, default=uuid4)
    owner_id    = Column(UUID, ForeignKey("account.id"), nullable=False)
    parent_id   = Column(UUID, ForeignKey("folder.id"), nullable=True)  # nested folders
    is_public   = Column(Boolean, default=False, nullable=False)
    # ... other fields
```

**Trade-off note:** Using `is_public` on the model is simpler and indexed easily. A wildcard `AccessGrant` row would let you track _who_ made it public and when, but adds join complexity to every permission check. `is_public` is the right default — you can always log changes via an audit table if needed.

---

# 6. Access Resolution Logic

When checking if Account X can view Video V owned by Account O:

```
1. Is V.is_public? → ALLOW (skip all further checks)
2. Is X blocked by O? (AccountBlock where blocker=O, blocked=X) → DENY
3. Is X == O? → ALLOW (owners always have access)
4. Is X an admin? → ALLOW
5. Is X a representative of O? → check AccessGrant for representatives
6. Does an AccessGrant exist where:
     owner_id = O
     AND (target_account_id = X OR target_group_id IN (X's group memberships))
     AND (video_id = V.id OR folder_id = V.folder_id OR (video_id IS NULL AND folder_id IS NULL))
     AND can_view = true
     AND (granted_until IS NULL OR granted_until > NOW())
   → ALLOW
7. Default → DENY
```

**Specificity rule:** More specific grants (video-level) should take precedence over broader ones (folder or global). Resolve in order: video → folder → global.

---

# 7. Enums

```python
class AccountType(str, Enum):
    ADMIN        = "admin"
    USER         = "user"
    TEAM         = "team"
    ORGANISATION = "organisation"


class RelationshipType(str, Enum):
    FRIEND         = "friend"
    MEMBER         = "member"
    REPRESENTATIVE = "representative"
    FOLLOWER       = "follower"
    INDIVIDUAL     = "individual"


class GrantedTo(str, Enum):
    EVERYONE = "everyone"
    ACCOUNT  = "account"
    GROUP    = "group"
```

> **Note:** Always use `class MyEnum(str, Enum)` — not just `class MyEnum(str)`. The latter gives you a plain class with string attributes, not a proper Python enum with iteration, validation, or SQLAlchemy integration.

---

# 8. Open TODOs

|Item|Recommendation|
|---|---|
|Representative visibility|Create an `AccessGrant` with `relationship_type = "representative"` targeting the representative's account or a "representatives" group. Grant `can_view`, `can_label` etc. as needed.|
|Folder inheritance|Decide: does a grant on a folder automatically apply to all videos inside? Recommended: yes, handle in resolution logic (step 6 above).|
|Audit trail|Consider an `AccessGrantHistory` table or Postgres audit trigger if you need to know who changed what and when.|
|Block propagation|Does blocking an account also block all their representatives? Probably yes — decide early.|
|Group-level blocks|Do you want to block entire groups, or only individual accounts? Currently only individual accounts are blockable.|