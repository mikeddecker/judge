# 🖋 Rename

```python
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.alter_column(
        'your_table_name',
        'labeldate',
        new_column_name='createdAt'
    )

def downgrade():
    op.alter_column(
        'your_table_name',
        'createdAt',
        new_column_name='labeldate'
    )

```

# ⚡ Bonus: rename + change properties

If you also changed nullable/default:

```python
op.alter_column(
    'your_table_name',
    'labeldate',
    new_column_name='createdAt',
    existing_type=sa.DateTime(),
    nullable=False
)
```
