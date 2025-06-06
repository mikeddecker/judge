"""video competitioninfo

Revision ID: 049dd53c8894
Revises: 8d01cde2bed1
Create Date: 2025-05-13 20:47:57.086006

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '049dd53c8894'
down_revision = '8d01cde2bed1'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('CompetitionInfo',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('info', sa.String(length=255), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('Videos', schema=None) as batch_op:
        batch_op.add_column(sa.Column('competition', sa.Integer(), nullable=True))
        batch_op.create_foreign_key(None, 'CompetitionInfo', ['competition'], ['id'])

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('Videos', schema=None) as batch_op:
        batch_op.drop_constraint(None, type_='foreignkey')
        batch_op.drop_column('competition')

    op.drop_table('CompetitionInfo')
    # ### end Alembic commands ###
