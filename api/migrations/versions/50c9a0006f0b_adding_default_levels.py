"""adding default levels

Revision ID: 50c9a0006f0b
Revises: 79ebd221b252
Create Date: 2025-02-11 15:28:38.484516

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '50c9a0006f0b'
down_revision = '79ebd221b252'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('Skillinfo_DoubleDutch_Skill', schema=None) as batch_op:
        batch_op.add_column(sa.Column('level', sa.String(length=7), nullable=False))

    with op.batch_alter_table('Skillinfo_DoubleDutch_Turner', schema=None) as batch_op:
        batch_op.add_column(sa.Column('level', sa.String(length=7), nullable=False))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('Skillinfo_DoubleDutch_Turner', schema=None) as batch_op:
        batch_op.drop_column('level')

    with op.batch_alter_table('Skillinfo_DoubleDutch_Skill', schema=None) as batch_op:
        batch_op.drop_column('level')

    # ### end Alembic commands ###
