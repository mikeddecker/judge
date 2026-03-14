"""add permissions: AccountCapability, Group, GroupMembership, AccessGrant, AccountBlock

Revision ID: c4d5e6f7a8b9
Revises: 1917fc57e9ec
Create Date: 2026-03-14 23:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

revision = 'c4d5e6f7a8b9'
down_revision = '1917fc57e9ec'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'AccountCapabilities',
        sa.Column('account_id', sa.BINARY(16), nullable=False),
        sa.Column('can_upload_video', sa.Boolean(), nullable=False),
        sa.Column('can_edit_video', sa.Boolean(), nullable=False),
        sa.Column('can_label_video', sa.Boolean(), nullable=False),
        sa.Column('can_see_video_actions', sa.Boolean(), nullable=False),
        sa.Column('can_see_tags', sa.Boolean(), nullable=False),
        sa.Column('can_edit_tags', sa.Boolean(), nullable=False),
        sa.Column('can_see_video_tags', sa.Boolean(), nullable=False),
        sa.Column('can_see_video_labels', sa.Boolean(), nullable=False),
        sa.Column('can_train_model', sa.Boolean(), nullable=False),
        sa.Column('can_export_model', sa.Boolean(), nullable=False),
        sa.Column('can_manage_members', sa.Boolean(), nullable=False),
        sa.Column('can_manage_representatives', sa.Boolean(), nullable=False),
        sa.Column('can_invite_users', sa.Boolean(), nullable=False),
        sa.Column('max_video_uploads', sa.Integer(), nullable=False),
        sa.Column('max_video_size_mb', sa.Integer(), nullable=False),
        sa.Column('max_video_duration_seconds', sa.Integer(), nullable=False),
        sa.Column('max_storage_gb', sa.Integer(), nullable=False),
        sa.Column('granted_by', sa.BINARY(16), nullable=False),
        sa.Column('granted_at', sa.DateTime(), nullable=False),
        sa.Column('granted_until', sa.DateTime(), nullable=True),
        sa.Column('granted_reason', sa.Text(), nullable=True),
        sa.Column('id', sa.BINARY(16), nullable=False),
        sa.Column('createdAt', sa.DateTime(), nullable=False),
        sa.Column('updatedAt', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['account_id'], ['Accounts.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['granted_by'], ['Accounts.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('account_id'),
    )

    op.create_table(
        'Groups',
        sa.Column('owner_id', sa.BINARY(16), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('id', sa.BINARY(16), nullable=False),
        sa.Column('createdAt', sa.DateTime(), nullable=False),
        sa.Column('updatedAt', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['owner_id'], ['Accounts.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table(
        'GroupMemberships',
        sa.Column('group_id', sa.BINARY(16), nullable=False),
        sa.Column('account_id', sa.BINARY(16), nullable=False),
        sa.Column('added_by', sa.BINARY(16), nullable=False),
        sa.Column('added_at', sa.DateTime(), nullable=False),
        sa.Column('id', sa.BINARY(16), nullable=False),
        sa.Column('createdAt', sa.DateTime(), nullable=False),
        sa.Column('updatedAt', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['account_id'], ['Accounts.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['added_by'], ['Accounts.id']),
        sa.ForeignKeyConstraint(['group_id'], ['Groups.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('group_id', 'account_id', name='_group_account_unique'),
    )

    op.create_table(
        'AccessGrants',
        sa.Column('owner_id', sa.BINARY(16), nullable=False),
        sa.Column('granted_to', sa.Enum('everyone', 'account', 'group', name='granted_to_enum'), nullable=False),
        sa.Column('target_account_id', sa.BINARY(16), nullable=True),
        sa.Column('target_group_id', sa.BINARY(16), nullable=True),
        sa.Column('video_id', sa.BINARY(16), nullable=True),
        sa.Column('folder_id', sa.BINARY(16), nullable=True),
        sa.Column('can_view', sa.Boolean(), nullable=False),
        sa.Column('can_comment', sa.Boolean(), nullable=False),
        sa.Column('can_label', sa.Boolean(), nullable=False),
        sa.Column('can_download', sa.Boolean(), nullable=False),
        sa.Column('relationship_type', sa.Enum('friend', 'member', 'representative', 'follower', 'individual', name='relationship_type_enum'), nullable=True),
        sa.Column('granted_by', sa.BINARY(16), nullable=False),
        sa.Column('granted_at', sa.DateTime(), nullable=False),
        sa.Column('granted_until', sa.DateTime(), nullable=True),
        sa.Column('granted_reason', sa.Text(), nullable=True),
        sa.Column('id', sa.BINARY(16), nullable=False),
        sa.Column('createdAt', sa.DateTime(), nullable=False),
        sa.Column('updatedAt', sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "(granted_to = 'everyone' AND target_account_id IS NULL AND target_group_id IS NULL)"
            " OR (granted_to = 'account' AND target_account_id IS NOT NULL AND target_group_id IS NULL)"
            " OR (granted_to = 'group' AND target_group_id IS NOT NULL AND target_account_id IS NULL)",
            name='ck_access_grant_target',
        ),
        sa.CheckConstraint(
            'NOT (video_id IS NOT NULL AND folder_id IS NOT NULL)',
            name='ck_access_grant_scope',
        ),
        sa.ForeignKeyConstraint(['folder_id'], ['Folders.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['granted_by'], ['Accounts.id']),
        sa.ForeignKeyConstraint(['owner_id'], ['Accounts.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['target_account_id'], ['Accounts.id']),
        sa.ForeignKeyConstraint(['target_group_id'], ['Groups.id']),
        sa.ForeignKeyConstraint(['video_id'], ['Videos.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table(
        'AccountBlocks',
        sa.Column('blocker_id', sa.BINARY(16), nullable=False),
        sa.Column('blocked_id', sa.BINARY(16), nullable=False),
        sa.Column('blocked_at', sa.DateTime(), nullable=False),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('id', sa.BINARY(16), nullable=False),
        sa.Column('createdAt', sa.DateTime(), nullable=False),
        sa.Column('updatedAt', sa.DateTime(), nullable=False),
        sa.CheckConstraint('blocker_id != blocked_id', name='ck_block_no_self'),
        sa.ForeignKeyConstraint(['blocked_id'], ['Accounts.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['blocker_id'], ['Accounts.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('blocker_id', 'blocked_id', name='_blocker_blocked_unique'),
    )


def downgrade():
    op.drop_table('AccountBlocks')
    op.drop_table('AccessGrants')
    op.drop_table('GroupMemberships')
    op.drop_table('Groups')
    op.drop_table('AccountCapabilities')
    op.execute('DROP TYPE IF EXISTS granted_to_enum')
    op.execute('DROP TYPE IF EXISTS relationship_type_enum')
