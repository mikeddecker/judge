from datetime import datetime
from uuid import UUID

from repository.permissionRepo import PermissionRepo

CAPABILITY_BOOL_FIELDS = (
    'can_upload_video', 'can_edit_video', 'can_label_video',
    'can_see_video_actions', 'can_see_tags', 'can_edit_tags',
    'can_see_video_tags', 'can_see_video_labels', 'can_train_model',
    'can_export_model', 'can_manage_members', 'can_manage_representatives',
    'can_invite_users',
)
CAPABILITY_INT_FIELDS = (
    'max_video_uploads', 'max_video_size_mb',
    'max_video_duration_seconds', 'max_storage_gb',
)

VALID_GRANTED_TO = ('everyone', 'account', 'group')
VALID_RELATIONSHIP_TYPES = ('friend', 'member', 'representative', 'follower', 'individual')


class PermissionService:
    """Business logic for permissions: capabilities, groups, grants, blocks."""

    # ── Access resolution ────────────────────────────────────────────────────

    @staticmethod
    def can_view_content(
        requester_id: UUID,
        owner_id: UUID,
        is_public: bool = False,
        video_id: UUID = None,
        folder_id: UUID = None,
    ) -> bool:
        """7-step access resolution (see documentation/features/AI-Judge.Permissions.md).

        1. Public content → ALLOW
        2. Blocked by owner → DENY
        3. Requester is owner → ALLOW
        4. Valid AccessGrant exists → ALLOW
        5. Default → DENY
        """
        if is_public:
            return True
        if PermissionRepo.is_blocked(owner_id, requester_id):
            return False
        if requester_id == owner_id:
            return True
        group_ids = PermissionRepo.get_account_group_ids(requester_id)
        return PermissionRepo.find_view_grants(
            requester_id_bytes=requester_id.bytes,
            owner_id_bytes=owner_id.bytes,
            group_id_bytes_list=group_ids,
            video_id_bytes=video_id.bytes if video_id else None,
            folder_id_bytes=folder_id.bytes if folder_id else None,
        )

    # ── AccountCapability ────────────────────────────────────────────────────

    @staticmethod
    def get_capability(account_id: UUID) -> dict:
        """Return capability dict for *account_id*, or None."""
        cap = PermissionRepo.get_capability(account_id)
        return cap.to_dict() if cap else None

    @staticmethod
    def set_capability(account_id: UUID, granted_by: UUID, fields: dict) -> dict:
        """Create or update capabilities for *account_id*.

        Only recognised field names are applied; unknown keys are ignored.
        Returns updated capability dict.
        """
        allowed = {**{k: fields[k] for k in CAPABILITY_BOOL_FIELDS if k in fields},
                   **{k: fields[k] for k in CAPABILITY_INT_FIELDS if k in fields}}
        if 'granted_until' in fields:
            allowed['granted_until'] = fields['granted_until']
        if 'granted_reason' in fields:
            allowed['granted_reason'] = fields['granted_reason']
        cap = PermissionRepo.upsert_capability(account_id, granted_by, **allowed)
        return cap.to_dict()

    # ── Groups ───────────────────────────────────────────────────────────────

    @staticmethod
    def create_group(owner_id: UUID, name: str, description: str = None) -> dict:
        """Create a group account (accountType='group'). Returns group dict."""
        if not name or not name.strip():
            return {'success': False, 'message': 'Group name is required'}
        group = PermissionRepo.create_group(owner_id, name.strip(), description)
        return {'success': True, 'group': _group_dict(group)}

    @staticmethod
    def list_groups(owner_id: UUID) -> list:
        """Return list of group dicts owned by *owner_id*."""
        return [_group_dict(g) for g in PermissionRepo.list_groups_by_owner(owner_id)]

    @staticmethod
    def delete_group(owner_id: UUID, group_id: UUID) -> dict:
        """Delete a group account if *owner_id* owns it."""
        group = PermissionRepo.get_group(group_id)
        if not group:
            return {'success': False, 'message': 'Group not found'}
        if group.owner_id != owner_id.bytes:
            return {'success': False, 'message': 'Forbidden'}
        PermissionRepo.delete_group(group_id)
        return {'success': True, 'message': 'Group deleted'}

    @staticmethod
    def add_member(owner_id: UUID, group_id: UUID, account_id: UUID) -> dict:
        """Add *account_id* to *group_id* if *owner_id* owns the group."""
        group = PermissionRepo.get_group(group_id)
        if not group:
            return {'success': False, 'message': 'Group not found'}
        if group.owner_id != owner_id.bytes:
            return {'success': False, 'message': 'Forbidden'}
        try:
            membership = PermissionRepo.add_member(group_id, account_id, owner_id)
            return {'success': True, 'membership': membership.to_dict()}
        except Exception:
            return {'success': False, 'message': 'Member already in group or invalid account'}

    @staticmethod
    def remove_member(owner_id: UUID, group_id: UUID, account_id: UUID) -> dict:
        """Remove *account_id* from *group_id*."""
        group = PermissionRepo.get_group(group_id)
        if not group:
            return {'success': False, 'message': 'Group not found'}
        if group.owner_id != owner_id.bytes:
            return {'success': False, 'message': 'Forbidden'}
        removed = PermissionRepo.remove_member(group_id, account_id)
        if not removed:
            return {'success': False, 'message': 'Member not found'}
        return {'success': True, 'message': 'Member removed'}

    # ── AccessGrants ─────────────────────────────────────────────────────────

    @staticmethod
    def create_grant(owner_id: UUID, data: dict) -> dict:
        """Create an access grant. *owner_id* is the content owner."""
        granted_to = data.get('granted_to')
        if granted_to not in VALID_GRANTED_TO:
            return {'success': False, 'message': f'granted_to must be one of {VALID_GRANTED_TO}'}

        target_account_id = _parse_uuid(data.get('target_account_id'))
        target_group_id = _parse_uuid(data.get('target_group_id'))
        video_id = _parse_uuid(data.get('video_id'))
        folder_id = _parse_uuid(data.get('folder_id'))
        relationship_type = data.get('relationship_type')
        if relationship_type and relationship_type not in VALID_RELATIONSHIP_TYPES:
            return {'success': False, 'message': f'relationship_type must be one of {VALID_RELATIONSHIP_TYPES}'}

        granted_until_raw = data.get('granted_until')
        granted_until = None
        if granted_until_raw:
            try:
                granted_until = datetime.fromisoformat(granted_until_raw)
            except (ValueError, TypeError):
                return {'success': False, 'message': 'Invalid granted_until format (ISO 8601 expected)'}

        try:
            grant = PermissionRepo.create_grant(
                owner_id=owner_id,
                granted_by=owner_id,
                granted_to=granted_to,
                target_account_id=target_account_id,
                target_group_id=target_group_id,
                video_id=video_id,
                folder_id=folder_id,
                can_view=bool(data.get('can_view', False)),
                can_comment=bool(data.get('can_comment', False)),
                can_label=bool(data.get('can_label', False)),
                can_download=bool(data.get('can_download', False)),
                relationship_type=relationship_type,
                granted_until=granted_until,
                granted_reason=data.get('granted_reason'),
            )
            return {'success': True, 'grant': grant.to_dict()}
        except Exception as exc:
            return {'success': False, 'message': str(exc)}

    @staticmethod
    def list_grants(owner_id: UUID) -> list:
        """Return list of grant dicts for *owner_id*."""
        return [g.to_dict() for g in PermissionRepo.list_grants_by_owner(owner_id)]

    @staticmethod
    def revoke_grant(owner_id: UUID, grant_id: UUID) -> dict:
        """Revoke a grant if *owner_id* owns it."""
        grant = PermissionRepo.get_grant(grant_id)
        if not grant:
            return {'success': False, 'message': 'Grant not found'}
        if grant.owner_id != owner_id.bytes:
            return {'success': False, 'message': 'Forbidden'}
        PermissionRepo.delete_grant(grant_id)
        return {'success': True, 'message': 'Grant revoked'}

    # ── AccountBlocks ────────────────────────────────────────────────────────

    @staticmethod
    def block_account(blocker_id: UUID, blocked_id: UUID, reason: str = None) -> dict:
        """Block *blocked_id*."""
        if blocker_id == blocked_id:
            return {'success': False, 'message': 'Cannot block yourself'}
        if PermissionRepo.is_blocked(blocker_id, blocked_id):
            return {'success': False, 'message': 'Already blocked'}
        block = PermissionRepo.create_block(blocker_id, blocked_id, reason)
        return {'success': True, 'block': block.to_dict()}

    @staticmethod
    def unblock_account(blocker_id: UUID, blocked_id: UUID) -> dict:
        """Unblock *blocked_id*."""
        removed = PermissionRepo.remove_block(blocker_id, blocked_id)
        if not removed:
            return {'success': False, 'message': 'Block not found'}
        return {'success': True, 'message': 'Account unblocked'}

    @staticmethod
    def list_blocks(blocker_id: UUID) -> list:
        """Return list of block dicts for *blocker_id*."""
        return [b.to_dict() for b in PermissionRepo.list_blocks(blocker_id)]


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_uuid(value) -> UUID:
    """Parse UUID from string, or return None."""
    if not value:
        return None
    try:
        return UUID(str(value))
    except (ValueError, AttributeError):
        return None


def _group_dict(group) -> dict:
    """Serialize a group Account (accountType='group') to a plain dict."""
    from uuid import UUID as _UUID
    owner_id_str = None
    if group.owner_id is not None:
        if isinstance(group.owner_id, bytes):
            owner_id_str = str(_UUID(bytes=group.owner_id))
        else:
            owner_id_str = str(group.owner_id)
    return {
        'id': group.uuid_str(),
        'name': group.firstName,
        'description': group.lastName or None,
        'owner_id': owner_id_str,
        'createdAt': group.createdAt.isoformat() if group.createdAt else None,
    }
