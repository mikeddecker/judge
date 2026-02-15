Notes about migrations: #TODO
(first info below from chatGPT.)

Switching an existing MySQL PK from `INT` → `UUID` in a Flask-SQLAlchemy + Alembic setup is a **data migration + schema migration**. The safe path is “add → backfill → switch → clean up”.

Below is the practical pattern used in production.

---

## ✅ 1) Decide how to store the UUID

For MySQL, best practice is:

**BINARY(16)** (fast, compact, indexed well)  
instead of  
CHAR(36) (human readable but slower & bigger)

In SQLAlchemy:

```python
import uuid
from sqlalchemy.dialects.mysql import BINARY

def generate_uuid():
    return uuid.uuid4().bytes  # 16-byte binary

class Video(db.Model):
    __tablename__ = "Videos"
    id = db.Column(BINARY(16), primary_key=True, default=generate_uuid)
```

If you want readable UUIDs in Python, convert when serializing.

❗First add uuid -> change FK's everywhere


---

## ✅ 2) Migration strategy (no downtime friendly)

### Step A — Add new UUID column (keep int id temporarily)

Alembic migration:

```python
def upgrade():
    op.add_column(
        "Videos",
        sa.Column("uuid", sa.BINARY(16), nullable=True)
    )
```

---

### Step B — Backfill existing rows

Use Python in migration (safer than SQL for UUID logic):

```python
import uuid
from sqlalchemy.sql import table, column
import sqlalchemy as sa

videos = table("Videos", column("id"), column("uuid"))

def upgrade():
    conn = op.get_bind()
    rows = conn.execute(sa.select(videos.c.id)).fetchall()

    for r in rows:
        conn.execute(
            videos.update()
            .where(videos.c.id == r.id)
            .values(uuid=uuid.uuid4().bytes)
        )
```

---

### Step C — Make UUID required + unique

```python
op.alter_column("Videos", "uuid", nullable=False)
op.create_unique_constraint("uq_videos_uuid", "Videos", ["uuid"])
```

---

### Step D — Switch primary key

⚠️ This is the only risky step. Ensure no FK still points to int id.

```python
op.drop_constraint("PRIMARY", "Videos", type_="primary")
op.create_primary_key("pk_videos", "Videos", ["uuid"])
```

If foreign keys exist:

1. Add UUID FK columns to child tables
    
2. Backfill them
    
3. Switch FK constraints
    
4. Then drop old int PK
    

---

### Step E — Drop old id column (optional)

```python
op.drop_column("Videos", "id")
op.alter_column("Videos", "uuid", new_column_name="id")
```

Now your UUID is the official PK.

---

## ❓ Do inserts need to handle double insert risk?

Short answer: **No special logic required if UUID is the PK.**

Why:

- UUID collisions are astronomically unlikely
    
- The database enforces uniqueness
    
- Normal insert semantics apply
    

Typical insert:

```python
video = Video(...)
db.session.add(video)
db.session.commit()
```

That’s it.

---

## ✅ When you _would_ handle duplicates

Only if you want **idempotent inserts** (same logical record shouldn’t be created twice). Then you choose one:

### Option A — Natural unique constraint

Example: `(user_id, external_source_id)` unique index

### Option B — Upsert pattern

MySQL:

```python
from sqlalchemy.dialects.mysql import insert

stmt = insert(Video).values(...)
stmt = stmt.on_duplicate_key_update(updated_at=sa.func.now())
db.session.execute(stmt)
```

Use this if your app might retry requests.

---

## ⚠️ Gotchas people hit

- MySQL requires dropping PK before creating new one
    
- All foreign keys must be migrated first
    
- Alembic autogenerate will NOT handle this automatically
    
- BINARY UUID must use `.bytes`, not string
    
- If using replication, do migration in phases
    

---

## 💡 Real-world recommendation

If this is a live system:

**Run in 2 deployments**  
1️⃣ Add UUID + backfill + app writes both ids  
2️⃣ Switch PK + remove int id

Zero downtime, zero drama.

---

If you want, paste your full model + FK relationships and I’ll map the exact Alembic migrations for your schema.