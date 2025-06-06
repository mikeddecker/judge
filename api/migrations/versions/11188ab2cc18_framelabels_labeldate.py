"""framelabels labeldate

Revision ID: 11188ab2cc18
Revises: 05402ab31644
Create Date: 2025-04-08 20:29:44.095138

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '11188ab2cc18'
down_revision = '05402ab31644'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('FrameLabels', schema=None) as batch_op:
        batch_op.add_column(sa.Column('labeldate', sa.DateTime(), nullable=True))

    with op.batch_alter_table('Predictions_Localization', schema=None) as batch_op:
        batch_op.alter_column('segmentSplitValue',
               existing_type=mysql.FLOAT(),
               nullable=True)

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('Predictions_Localization', schema=None) as batch_op:
        batch_op.alter_column('segmentSplitValue',
               existing_type=mysql.FLOAT(),
               nullable=False)

    with op.batch_alter_table('FrameLabels', schema=None) as batch_op:
        batch_op.drop_column('labeldate')

    # ### end Alembic commands ###
