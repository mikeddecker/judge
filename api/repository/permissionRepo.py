from datetime import datetime
from uuid import UUID

from repository.db import db
from repository.models import (
    AccessGrant, AccountBlock, AccountCapability, Group, GroupMembership
)


class PermissionRepo:
    """Data-access layer for the permissions system."""

    # ── AccountCapability ────────────────────────────────────────────────────

    @staticmethod
    def get_capability(account_id: UUID) -> AccountCapability:
        """Return the capability row for *account_id*, or None."""
        return AccountCapability.query.filter_by(account_id=account_id.bytes).first()

    @staticmethod
    def upsert_capability(account_id: UUID, granted_by: UUID, **fields) -> AccountCapability:
        """Create or fully update the capability row for *account_id*."""
        cap = AccountCapability.query.filter_by(account_id=account_id.bytes).first()
        if cap is None:
            cap = AccountCapability(
                account_id=account_id.bytes,
                granted_by=granted_by.bytes,
                granted_at=datetime.now(),
            )
            db.session.add(cap)
        else:
            cap.granted_by = granted_by.bytes
            cap.granted_at = datetime.now()
        for key, value in fields.items():
            setattr(cap, key, value)
        db.session.commit()
        return cap

    # ── Group ────────────────────────────────────────────────────────────────

    @staticmethod
    def create_group(owner_id: UUID, name: str, description: str = None) -> Group:
        """Create a new group owned by *owner_id*."""
        group = Group(
            owner_id=owner_id.bytes,
            name=name,
            description=description,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
        )
        db.session.add(group)
        db.session.commit()
        return group

    @staticmethod
    def get_group(group_id: UUID) -> Group:
        """Return group by ID or None."""
        return Group.query.get(group_id.bytes)

    @staticmethod
    def list_groups_by_owner(owner_id: UUID) -> list:
        """Return all groups owned by *owner_id*."""
        return Group.query.filter_by(owner_id=owner_id.bytes).all()

    @staticmethod
    def delete_group(group_id: UUID) -> bool:
        """Delete a group. Returns True if deleted, False if not found."""
        group = Group.query.get(group_id.bytes)
        if not group:
            return False
        db.session.delete(group)
        db.session.commit()
        return True

    # ── GroupMembership ──────────────────────────────────────────────────────

    @staticmethod
    def add_member(group_id: UUID, account_id: UUID, added_by: UUID) -> GroupMembership:
        """Add *account_id* to *group_id*."""
        membership = GroupMembership(
            group_id=group_id.bytes,
            account_id=account_id.bytes,
            added_by=added_by.bytes,
            added_at=datetime.now(),
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
        )
        db.session.add(membership)
        db.session.commit()
        return membership

    @staticmethod
    def remove_member(group_id: UUID, account_id: UUID) -> bool:
        """Remove *account_id* from *group_id*. Returns True if removed."""
        membership = GroupMembership.query.filter_by(
            group_id=group_id.bytes, account_id=account_id.bytes
        ).first()
        if not membership:
            return False
        db.session.delete(membership)
        db.session.commit()
        return True

    @staticmethod
    def get_account_group_ids(account_id: UUID) -> list:
        """Return list of raw group_id bytes for *account_id*."""
        memberships = GroupMembership.query.filter_by(account_id=account_id.bytes).all()
        return [m.group_id for m in memberships]

    # ── AccessGrant ──────────────────────────────────────────────────────────

    @staticmethod
    def create_grant(
        owner_id: UUID,
        granted_by: UUID,
        granted_to: str,
        target_account_id: UUID = None,
        target_group_id: UUID = None,
        video_id: UUID = None,
        folder_id: UUID = None,
        can_view: bool = False,
        can_comment: bool = False,
        can_label: bool = False,
        can_download: bool = False,
        relationship_type: str = None,
        granted_until: datetime = None,
        granted_reason: str = None,
    ) -> AccessGrant:
        """Create a new access grant."""
        grant = AccessGrant(
            owner_id=owner_id.bytes,
            granted_by=granted_by.bytes,
            granted_to=granted_to,
            target_account_id=target_account_id.bytes if target_account_id else None,
            target_group_id=target_group_id.bytes if target_group_id else None,
            video_id=video_id.bytes if video_id else None,
            folder_id=folder_id.bytes if folder_id else None,
            can_view=can_view,
            can_comment=can_comment,
            can_label=can_label,
            can_download=can_download,
            relationship_type=relationship_type,
            granted_at=datetime.now(),
            granted_until=granted_until,
            granted_reason=granted_reason,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
        )
        db.session.add(grant)
        db.session.commit()
        return grant

    @staticmethod
    def get_grant(grant_id: UUID) -> AccessGrant:
        """Return grant by ID or None."""
        return AccessGrant.query.get(grant_id.bytes)

    @staticmethod
    def list_grants_by_owner(owner_id: UUID) -> list:
        """Return all grants created by *owner_id*."""
        return AccessGrant.query.filter_by(owner_id=owner_id.bytes).all()

    @staticmethod
    def delete_grant(grant_id: UUID) -> bool:
        """Delete a grant. Returns True if deleted."""
        grant = AccessGrant.query.get(grant_id.bytes)
        if not grant:
            return False
        db.session.delete(grant)
        db.session.commit()
        return True

    @staticmethod
    def find_view_grants(
        requester_id_bytes: bytes,
        owner_id_bytes: bytes,
        group_id_bytes_list: list,
        video_id_bytes: bytes = None,
        folder_id_bytes: bytes = None,
    ) -> bool:
        """Return True if a valid can_view grant exists for requester on owner's content."""
        now = datetime.now()
        target_filter = db.or_(
            AccessGrant.granted_to == 'everyone',
            db.and_(
                AccessGrant.granted_to == 'account',
                AccessGrant.target_account_id == requester_id_bytes,
            ),
            *(
                [db.and_(
                    AccessGrant.granted_to == 'group',
                    AccessGrant.target_group_id.in_(group_id_bytes_list),
                )]
                if group_id_bytes_list else []
            ),
        )

        if video_id_bytes is not None:
            scope_filter = db.or_(
                AccessGrant.video_id == video_id_bytes,
                db.and_(
                    AccessGrant.folder_id == folder_id_bytes,
                    AccessGrant.folder_id.isnot(None),
                ) if folder_id_bytes else db.false(),
                db.and_(AccessGrant.video_id.is_(None), AccessGrant.folder_id.is_(None)),
            )
        elif folder_id_bytes is not None:
            scope_filter = db.or_(
                AccessGrant.folder_id == folder_id_bytes,
                db.and_(AccessGrant.video_id.is_(None), AccessGrant.folder_id.is_(None)),
            )
        else:
            scope_filter = db.and_(AccessGrant.video_id.is_(None), AccessGrant.folder_id.is_(None))

        return AccessGrant.query.filter(
            AccessGrant.owner_id == owner_id_bytes,
            AccessGrant.can_view.is_(True),
            db.or_(AccessGrant.granted_until.is_(None), AccessGrant.granted_until > now),
            target_filter,
            scope_filter,
        ).first() is not None

    # ── AccountBlock ─────────────────────────────────────────────────────────

    @staticmethod
    def create_block(blocker_id: UUID, blocked_id: UUID, reason: str = None) -> AccountBlock:
        """Block *blocked_id* from *blocker_id*'s content."""
        block = AccountBlock(
            blocker_id=blocker_id.bytes,
            blocked_id=blocked_id.bytes,
            blocked_at=datetime.now(),
            reason=reason,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
        )
        db.session.add(block)
        db.session.commit()
        return block

    @staticmethod
    def remove_block(blocker_id: UUID, blocked_id: UUID) -> bool:
        """Remove block. Returns True if removed."""
        block = AccountBlock.query.filter_by(
            blocker_id=blocker_id.bytes, blocked_id=blocked_id.bytes
        ).first()
        if not block:
            return False
        db.session.delete(block)
        db.session.commit()
        return True

    @staticmethod
    def list_blocks(blocker_id: UUID) -> list:
        """Return all blocks created by *blocker_id*."""
        return AccountBlock.query.filter_by(blocker_id=blocker_id.bytes).all()

    @staticmethod
    def is_blocked(blocker_id: UUID, blocked_id: UUID) -> bool:
        """Return True if *blocker_id* has blocked *blocked_id*."""
        return AccountBlock.query.filter_by(
            blocker_id=blocker_id.bytes, blocked_id=blocked_id.bytes
        ).first() is not None
