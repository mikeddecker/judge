"""Update folder rename path to name

Revision ID: bdb4b32df0c1
Revises: 5a7a952d3aaf
Create Date: 2024-12-07 09:09:19.558602

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = 'bdb4b32df0c1'
down_revision = '5a7a952d3aaf'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('Folders', schema=None) as batch_op:
        batch_op.add_column(sa.Column('name', sa.String(length=127), nullable=False))
        batch_op.drop_index('path')
        batch_op.create_unique_constraint(None, ['name'])
        batch_op.drop_column('path')

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('Folders', schema=None) as batch_op:
        batch_op.add_column(sa.Column('path', mysql.VARCHAR(length=127), nullable=False))
        batch_op.drop_constraint(None, type_='unique')
        batch_op.create_index('path', ['path'], unique=True)
        batch_op.drop_column('name')

    # ### end Alembic commands ###
