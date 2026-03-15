import { getApplicationJson, postApplicationJson, putApplicationJson, deleteApplicationJson } from './api'

const permissionService = {
  // ── Capabilities ──────────────────────────────────────────────────────────
  getCapability: (accountId) => {
    return getApplicationJson(`/capabilities/${accountId}`)
  },

  setCapability: (accountId, fields) => {
    return putApplicationJson(`/capabilities/${accountId}`, fields)
  },

  // ── Groups ────────────────────────────────────────────────────────────────
  listGroups: () => {
    return getApplicationJson('/groups')
  },

  createGroup: (name, description) => {
    return postApplicationJson('/groups', { name, description })
  },

  deleteGroup: (groupId) => {
    return deleteApplicationJson(`/groups/${groupId}`)
  },

  addMember: (groupId, accountId) => {
    return postApplicationJson(`/groups/${groupId}/members`, { account_id: accountId })
  },

  removeMember: (groupId, accountId) => {
    return deleteApplicationJson(`/groups/${groupId}/members`, { account_id: accountId })
  },

  // ── Access Grants ─────────────────────────────────────────────────────────
  listGrants: () => {
    return getApplicationJson('/access-grants')
  },

  createGrant: (payload) => {
    return postApplicationJson('/access-grants', payload)
  },

  revokeGrant: (grantId) => {
    return deleteApplicationJson(`/access-grants/${grantId}`)
  },

  // ── Blocks ────────────────────────────────────────────────────────────────
  listBlocks: () => {
    return getApplicationJson('/blocks')
  },

  blockAccount: (accountId, reason) => {
    return postApplicationJson('/blocks', { account_id: accountId, reason })
  },

  unblockAccount: (accountId) => {
    return deleteApplicationJson(`/blocks/${accountId}`)
  },
}

export default permissionService
