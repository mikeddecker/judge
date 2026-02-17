"""Default DomainObject class (id, createdAt, updatedAt)

Revision ID: 51b4519175e4
Revises: a84c473b4a00
Create Date: 2026-02-17 19:16:59.232055

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '51b4519175e4'
down_revision = 'a84c473b4a00'
branch_labels = None
depends_on = None

def upgrade():

    # ---- direct renames ----
    with op.batch_alter_table('FrameLabels') as batch_op:
        batch_op.alter_column('labeldatetime', new_column_name='createdAt', existing_type=sa.DateTime())
        batch_op.add_column(sa.Column('updatedAt', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')))

    with op.batch_alter_table('Jobs') as batch_op:
        batch_op.alter_column('request_time', new_column_name='createdAt', existing_type=sa.DateTime())
        batch_op.add_column(sa.Column('updatedAt', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')))

    # ---- creationDate / lastUpdated pattern ----
    for table in ['LayerComposition', 'LayerValues', 'Layers',
                  'TrainResults', 'TrainResultsEpoch']:
        with op.batch_alter_table(table) as batch_op:
            batch_op.alter_column('creationDate', new_column_name='createdAt', existing_type=sa.DateTime())
            batch_op.alter_column('lastUpdated', new_column_name='updatedAt', existing_type=sa.DateTime())

    # ---- Skills special names ----
    with op.batch_alter_table('Skills') as batch_op:
        batch_op.alter_column('labeldate', new_column_name='createdAt', existing_type=sa.DateTime())
        batch_op.alter_column('updated', new_column_name='updatedAt', existing_type=sa.DateTime())

    # ---- tables that had no timestamps before ----
    for table in [
        'CompetitionInfo', 'Folders', 'FrameLabelTypes',
        'Sources', 'TagGroups', 'Tags', 'Videos'
    ]:
        with op.batch_alter_table(table) as batch_op:
            batch_op.add_column(sa.Column('createdAt', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')))
            batch_op.add_column(sa.Column('updatedAt', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')))

    with op.batch_alter_table('TrainResults', schema=None) as batch_op:
        batch_op.drop_column('trainStart')

def downgrade():
    with op.batch_alter_table('TrainResults', schema=None) as batch_op:
        batch_op.add_column(sa.Column('trainStart', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')))

    op.execute("""
        UPDATE TrainResults
        SET trainStart = createdAt
    """)

    with op.batch_alter_table('FrameLabels') as batch_op:
        batch_op.alter_column('createdAt', new_column_name='labeldatetime', existing_type=sa.DateTime())
        batch_op.drop_column('updatedAt')

    with op.batch_alter_table('Jobs') as batch_op:
        batch_op.alter_column('createdAt', new_column_name='request_time', existing_type=sa.DateTime())
        batch_op.drop_column('updatedAt')

    for table in ['LayerComposition', 'LayerValues', 'Layers',
                  'TrainResults', 'TrainResultsEpoch']:
        with op.batch_alter_table(table) as batch_op:
            batch_op.alter_column('createdAt', new_column_name='creationDate', existing_type=sa.DateTime())
            batch_op.alter_column('updatedAt', new_column_name='lastUpdated', existing_type=sa.DateTime())

    with op.batch_alter_table('Skills') as batch_op:
        batch_op.alter_column('createdAt', new_column_name='labeldate', existing_type=sa.DateTime())
        batch_op.alter_column('updatedAt', new_column_name='updated', existing_type=sa.DateTime())

    for table in [
        'CompetitionInfo', 'Folders', 'FrameLabelTypes',
        'Sources', 'TagGroups', 'Tags', 'Videos'
    ]:
        with op.batch_alter_table(table) as batch_op:
            batch_op.drop_column('updatedAt')
            batch_op.drop_column('createdAt')


