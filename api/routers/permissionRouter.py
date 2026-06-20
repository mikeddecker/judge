from flask import request, session
from flask_restful import Resource
from services.permissionService import PermissionService
from uuid import UUID


def _parse_uuid(value) -> UUID:
    """Parse UUID from string/UUID, or return None."""
    if not value:
        return None
    try:
        return UUID(str(value))
    except (ValueError, AttributeError):
        return None


def _require_auth():
    """Return (account_id UUID, None) or (None, error response tuple)."""
    if 'account_id' not in session:
        return None, ({'success': False, 'message': 'Not authenticated'}, 401)
    account_id = session['account_id']
    if isinstance(account_id, bytes):
        account_id = UUID(bytes=account_id)
    elif not isinstance(account_id, UUID):
        account_id = UUID(str(account_id))
    return account_id, None


# ── Capabilities ─────────────────────────────────────────────────────────────

class CapabilityRouter(Resource):
    def get(self, account_id: str):
        """Get capabilities for an account"""
        caller_id, err = _require_auth()
        if err:
            return err
        target_id = _parse_uuid(account_id)
        if not target_id:
            return {'success': False, 'message': 'Invalid account_id'}, 400
        cap = PermissionService.get_capability(target_id)
        if cap is None:
            return {'success': False, 'message': 'No capabilities found'}, 404
        return {'success': True, 'capability': cap}, 200

    def put(self, account_id: str):
        """Set (upsert) capabilities for an account — admin operation"""
        caller_id, err = _require_auth()
        if err:
            return err
        target_id = _parse_uuid(account_id)
        if not target_id:
            return {'success': False, 'message': 'Invalid account_id'}, 400
        data = request.get_json() or {}
        cap = PermissionService.set_capability(target_id, caller_id, data)
        return {'success': True, 'capability': cap}, 200


# ── Groups ────────────────────────────────────────────────────────────────────

class GroupListRouter(Resource):
    def get(self):
        """List groups owned by the current account"""
        caller_id, err = _require_auth()
        if err:
            return err
        groups = PermissionService.list_groups(caller_id)
        return {'success': True, 'groups': groups}, 200

    def post(self):
        """Create a new group"""
        caller_id, err = _require_auth()
        if err:
            return err
        data = request.get_json() or {}
        result = PermissionService.create_group(
            caller_id,
            name=data.get('name', ''),
            description=data.get('description'),
        )
        status = 201 if result['success'] else 400
        return result, status


class GroupDetailRouter(Resource):
    def delete(self, group_id: str):
        """Delete a group"""
        caller_id, err = _require_auth()
        if err:
            return err
        gid = _parse_uuid(group_id)
        if not gid:
            return {'success': False, 'message': 'Invalid group_id'}, 400
        result = PermissionService.delete_group(caller_id, gid)
        status = 200 if result['success'] else (403 if result['message'] == 'Forbidden' else 404)
        return result, status


class GroupMemberRouter(Resource):
    def post(self, group_id: str):
        """Add a member to a group"""
        caller_id, err = _require_auth()
        if err:
            return err
        gid = _parse_uuid(group_id)
        if not gid:
            return {'success': False, 'message': 'Invalid group_id'}, 400
        data = request.get_json() or {}
        account_id = _parse_uuid(data.get('account_id'))
        if not account_id:
            return {'success': False, 'message': 'account_id is required'}, 400
        result = PermissionService.add_member(caller_id, gid, account_id)
        status = 201 if result['success'] else (403 if result['message'] == 'Forbidden' else 400)
        return result, status

    def delete(self, group_id: str):
        """Remove a member from a group"""
        caller_id, err = _require_auth()
        if err:
            return err
        gid = _parse_uuid(group_id)
        if not gid:
            return {'success': False, 'message': 'Invalid group_id'}, 400
        data = request.get_json() or {}
        account_id = _parse_uuid(data.get('account_id'))
        if not account_id:
            return {'success': False, 'message': 'account_id is required'}, 400
        result = PermissionService.remove_member(caller_id, gid, account_id)
        status = 200 if result['success'] else (403 if result['message'] == 'Forbidden' else 404)
        return result, status


# ── AccessGrants ─────────────────────────────────────────────────────────────

class AccessGrantListRouter(Resource):
    def get(self):
        """List access grants for the current account"""
        caller_id, err = _require_auth()
        if err:
            return err
        grants = PermissionService.list_grants(caller_id)
        return {'success': True, 'grants': grants}, 200

    def post(self):
        """Create an access grant"""
        caller_id, err = _require_auth()
        if err:
            return err
        data = request.get_json() or {}
        result = PermissionService.create_grant(caller_id, data)
        status = 201 if result['success'] else 400
        return result, status


class AccessGrantDetailRouter(Resource):
    def delete(self, grant_id: str):
        """Revoke an access grant"""
        caller_id, err = _require_auth()
        if err:
            return err
        gid = _parse_uuid(grant_id)
        if not gid:
            return {'success': False, 'message': 'Invalid grant_id'}, 400
        result = PermissionService.revoke_grant(caller_id, gid)
        status = 200 if result['success'] else (403 if result['message'] == 'Forbidden' else 404)
        return result, status


# ── Blocks ────────────────────────────────────────────────────────────────────

class BlockListRouter(Resource):
    def get(self):
        """List accounts blocked by the current account"""
        caller_id, err = _require_auth()
        if err:
            return err
        blocks = PermissionService.list_blocks(caller_id)
        return {'success': True, 'blocks': blocks}, 200

    def post(self):
        """Block an account"""
        caller_id, err = _require_auth()
        if err:
            return err
        data = request.get_json() or {}
        blocked_id = _parse_uuid(data.get('account_id'))
        if not blocked_id:
            return {'success': False, 'message': 'account_id is required'}, 400
        result = PermissionService.block_account(caller_id, blocked_id, data.get('reason'))
        status = 201 if result['success'] else 400
        return result, status


class BlockDetailRouter(Resource):
    def delete(self, account_id: str):
        """Unblock an account"""
        caller_id, err = _require_auth()
        if err:
            return err
        blocked_id = _parse_uuid(account_id)
        if not blocked_id:
            return {'success': False, 'message': 'Invalid account_id'}, 400
        result = PermissionService.unblock_account(caller_id, blocked_id)
        status = 200 if result['success'] else 404
        return result, status
