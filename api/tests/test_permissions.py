# -*- coding: utf-8 -*-
"""Unit tests for the permissions service layer (no DB required)."""

import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4, UUID

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uuid() -> UUID:
    return uuid4()


def _mock_cap(**overrides):
    """Return a mock AccountCapability dict."""
    defaults = {
        'id': str(_uuid()),
        'account_id': str(_uuid()),
        'can_upload_video': False,
        'can_edit_video': False,
        'can_label_video': False,
        'can_see_video_actions': False,
        'can_see_tags': False,
        'can_edit_tags': False,
        'can_see_video_tags': False,
        'can_see_video_labels': False,
        'can_train_model': False,
        'can_export_model': False,
        'can_manage_members': False,
        'can_manage_representatives': False,
        'can_invite_users': False,
        'max_video_uploads': 100,
        'max_video_size_mb': 2048,
        'max_video_duration_seconds': 120,
        'max_storage_gb': 20,
        'granted_by': str(_uuid()),
        'granted_at': '2026-01-01T00:00:00',
        'granted_until': None,
        'granted_reason': None,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# PermissionService._parse_uuid
# ---------------------------------------------------------------------------

class TestParseUUID(unittest.TestCase):
    def setUp(self):
        from services.permissionService import _parse_uuid
        self.parse = _parse_uuid

    def test_valid_string(self):
        uid = _uuid()
        self.assertEqual(self.parse(str(uid)), uid)

    def test_none_returns_none(self):
        self.assertIsNone(self.parse(None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(self.parse(''))

    def test_invalid_string_returns_none(self):
        self.assertIsNone(self.parse('not-a-uuid'))

    def test_uuid_object_returns_same(self):
        uid = _uuid()
        self.assertEqual(self.parse(uid), uid)


# ---------------------------------------------------------------------------
# PermissionService.can_view_content
# ---------------------------------------------------------------------------

class TestCanViewContent(unittest.TestCase):
    def setUp(self):
        from services.permissionService import PermissionService
        self.svc = PermissionService

    @patch('services.permissionService.PermissionRepo')
    def test_public_content_always_allowed(self, repo):
        self.assertTrue(self.svc.can_view_content(_uuid(), _uuid(), is_public=True))
        repo.is_blocked.assert_not_called()

    @patch('services.permissionService.PermissionRepo')
    def test_blocked_requester_denied(self, repo):
        owner = _uuid()
        requester = _uuid()
        repo.is_blocked.return_value = True
        self.assertFalse(
            self.svc.can_view_content(requester, owner, is_public=False)
        )
        repo.is_blocked.assert_called_once_with(owner, requester)

    @patch('services.permissionService.PermissionRepo')
    def test_owner_always_allowed(self, repo):
        uid = _uuid()
        repo.is_blocked.return_value = False
        self.assertTrue(self.svc.can_view_content(uid, uid, is_public=False))
        repo.find_view_grants.assert_not_called()

    @patch('services.permissionService.PermissionRepo')
    def test_valid_grant_allows(self, repo):
        owner = _uuid()
        requester = _uuid()
        repo.is_blocked.return_value = False
        repo.get_account_group_ids.return_value = []
        repo.find_view_grants.return_value = True
        self.assertTrue(self.svc.can_view_content(requester, owner))

    @patch('services.permissionService.PermissionRepo')
    def test_no_grant_denies(self, repo):
        owner = _uuid()
        requester = _uuid()
        repo.is_blocked.return_value = False
        repo.get_account_group_ids.return_value = []
        repo.find_view_grants.return_value = False
        self.assertFalse(self.svc.can_view_content(requester, owner))


# ---------------------------------------------------------------------------
# PermissionService.create_group / delete_group
# ---------------------------------------------------------------------------

class TestGroupService(unittest.TestCase):
    def setUp(self):
        from services.permissionService import PermissionService
        self.svc = PermissionService

    @patch('services.permissionService.PermissionRepo')
    def test_create_group_empty_name_fails(self, repo):
        result = self.svc.create_group(_uuid(), name='')
        self.assertFalse(result['success'])
        repo.create_group.assert_not_called()

    @patch('services.permissionService.PermissionRepo')
    def test_create_group_whitespace_name_fails(self, repo):
        result = self.svc.create_group(_uuid(), name='   ')
        self.assertFalse(result['success'])

    @patch('services.permissionService.PermissionRepo')
    def test_create_group_success(self, repo):
        owner = _uuid()
        mock_group = MagicMock()
        mock_group.uuid_str.return_value = str(_uuid())
        mock_group.firstName = 'Test'
        mock_group.lastName = ''
        mock_group.owner_id = owner.bytes
        mock_group.createdAt = None
        repo.create_group.return_value = mock_group
        result = self.svc.create_group(owner, name='Test')
        self.assertTrue(result['success'])
        repo.create_group.assert_called_once_with(owner, 'Test', None)

    @patch('services.permissionService.PermissionRepo')
    def test_delete_group_not_found(self, repo):
        repo.get_group.return_value = None
        result = self.svc.delete_group(_uuid(), _uuid())
        self.assertFalse(result['success'])

    @patch('services.permissionService.PermissionRepo')
    def test_delete_group_forbidden(self, repo):
        caller = _uuid()
        owner = _uuid()
        mock_group = MagicMock()
        mock_group.owner_id = owner.bytes  # different owner
        repo.get_group.return_value = mock_group
        result = self.svc.delete_group(caller, _uuid())
        self.assertFalse(result['success'])
        self.assertEqual(result['message'], 'Forbidden')

    @patch('services.permissionService.PermissionRepo')
    def test_delete_group_success(self, repo):
        caller = _uuid()
        mock_group = MagicMock()
        mock_group.owner_id = caller.bytes
        repo.get_group.return_value = mock_group
        result = self.svc.delete_group(caller, _uuid())
        self.assertTrue(result['success'])


# ---------------------------------------------------------------------------
# PermissionService.create_grant
# ---------------------------------------------------------------------------

class TestAccessGrantService(unittest.TestCase):
    def setUp(self):
        from services.permissionService import PermissionService
        self.svc = PermissionService

    @patch('services.permissionService.PermissionRepo')
    def test_invalid_granted_to_fails(self, repo):
        result = self.svc.create_grant(_uuid(), {'granted_to': 'invalid'})
        self.assertFalse(result['success'])

    @patch('services.permissionService.PermissionRepo')
    def test_invalid_relationship_type_fails(self, repo):
        result = self.svc.create_grant(_uuid(), {
            'granted_to': 'everyone',
            'relationship_type': 'unknown',
        })
        self.assertFalse(result['success'])

    @patch('services.permissionService.PermissionRepo')
    def test_invalid_granted_until_fails(self, repo):
        result = self.svc.create_grant(_uuid(), {
            'granted_to': 'everyone',
            'granted_until': 'not-a-date',
        })
        self.assertFalse(result['success'])

    @patch('services.permissionService.PermissionRepo')
    def test_create_grant_everyone_success(self, repo):
        mock_grant = MagicMock()
        mock_grant.to_dict.return_value = {'id': str(_uuid())}
        repo.create_grant.return_value = mock_grant
        result = self.svc.create_grant(_uuid(), {
            'granted_to': 'everyone',
            'can_view': True,
        })
        self.assertTrue(result['success'])

    @patch('services.permissionService.PermissionRepo')
    def test_revoke_grant_not_found(self, repo):
        repo.get_grant.return_value = None
        result = self.svc.revoke_grant(_uuid(), _uuid())
        self.assertFalse(result['success'])

    @patch('services.permissionService.PermissionRepo')
    def test_revoke_grant_forbidden(self, repo):
        caller = _uuid()
        other = _uuid()
        mock_grant = MagicMock()
        mock_grant.owner_id = other.bytes
        repo.get_grant.return_value = mock_grant
        result = self.svc.revoke_grant(caller, _uuid())
        self.assertFalse(result['success'])
        self.assertEqual(result['message'], 'Forbidden')

    @patch('services.permissionService.PermissionRepo')
    def test_revoke_grant_success(self, repo):
        caller = _uuid()
        mock_grant = MagicMock()
        mock_grant.owner_id = caller.bytes
        repo.get_grant.return_value = mock_grant
        result = self.svc.revoke_grant(caller, _uuid())
        self.assertTrue(result['success'])


# ---------------------------------------------------------------------------
# PermissionService.block / unblock
# ---------------------------------------------------------------------------

class TestBlockService(unittest.TestCase):
    def setUp(self):
        from services.permissionService import PermissionService
        self.svc = PermissionService

    @patch('services.permissionService.PermissionRepo')
    def test_cannot_block_self(self, repo):
        uid = _uuid()
        result = self.svc.block_account(uid, uid)
        self.assertFalse(result['success'])
        repo.create_block.assert_not_called()

    @patch('services.permissionService.PermissionRepo')
    def test_block_already_blocked(self, repo):
        blocker = _uuid()
        blocked = _uuid()
        repo.is_blocked.return_value = True
        result = self.svc.block_account(blocker, blocked)
        self.assertFalse(result['success'])

    @patch('services.permissionService.PermissionRepo')
    def test_block_success(self, repo):
        blocker = _uuid()
        blocked = _uuid()
        repo.is_blocked.return_value = False
        mock_block = MagicMock()
        mock_block.to_dict.return_value = {'id': str(_uuid())}
        repo.create_block.return_value = mock_block
        result = self.svc.block_account(blocker, blocked, reason='spam')
        self.assertTrue(result['success'])

    @patch('services.permissionService.PermissionRepo')
    def test_unblock_not_found(self, repo):
        repo.remove_block.return_value = False
        result = self.svc.unblock_account(_uuid(), _uuid())
        self.assertFalse(result['success'])

    @patch('services.permissionService.PermissionRepo')
    def test_unblock_success(self, repo):
        repo.remove_block.return_value = True
        result = self.svc.unblock_account(_uuid(), _uuid())
        self.assertTrue(result['success'])


if __name__ == '__main__':
    unittest.main()
