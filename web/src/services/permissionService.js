import { api } from './api'

const permissionService = {
  // ── Capabilities ──────────────────────────────────────────────────────────
  getCapability: async (accountId) => {
    const resp = await api.get(`/capabilities/${accountId}`)
    return resp.data
  },

  setCapability: async (accountId, fields) => {
    const resp = await api.put(`/capabilities/${accountId}`, fields)
    return resp.data
  },

  // ── Groups ────────────────────────────────────────────────────────────────
  listGroups: async () => {
    const resp = await api.get('/groups')
    return resp.data
  },

  createGroup: async (name, description) => {
    const resp = await api.post('/groups', { name, description })
    return resp.data
  },

  deleteGroup: async (groupId) => {
    const resp = await api.delete(`/groups/${groupId}`)
    return resp.data
  },

  addMember: async (groupId, accountId) => {
    const resp = await api.post(`/groups/${groupId}/members`, { account_id: accountId })
    return resp.data
  },

  removeMember: async (groupId, accountId) => {
    const resp = await api.delete(`/groups/${groupId}/members`, { data: { account_id: accountId } })
    return resp.data
  },

  // ── Access Grants ─────────────────────────────────────────────────────────
  listGrants: async () => {
    const resp = await api.get('/access-grants')
    return resp.data
  },

  createGrant: async (payload) => {
    const resp = await api.post('/access-grants', payload)
    return resp.data
  },

  revokeGrant: async (grantId) => {
    const resp = await api.delete(`/access-grants/${grantId}`)
    return resp.data
  },

  // ── Blocks ────────────────────────────────────────────────────────────────
  listBlocks: async () => {
    const resp = await api.get('/blocks')
    return resp.data
  },

  blockAccount: async (accountId, reason) => {
    const resp = await api.post('/blocks', { account_id: accountId, reason })
    return resp.data
  },

  unblockAccount: async (accountId) => {
    const resp = await api.delete(`/blocks/${accountId}`)
    return resp.data
  },
}

export default permissionService
