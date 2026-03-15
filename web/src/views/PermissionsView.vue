<template>
  <div class="permissions-container">
    <h1>Permissions</h1>

    <TabView>
      <!-- Groups -->
      <TabPanel header="Groups">
        <div class="section">
          <div class="section-header">
            <h2>My Groups</h2>
            <Button label="New Group" icon="pi pi-plus" size="small" @click="showCreateGroup = true" />
          </div>

          <DataTable :value="groups" :loading="loadingGroups" empty-message="No groups yet.">
            <Column field="name" header="Name" />
            <Column field="description" header="Description" />
            <Column field="createdAt" header="Created" />
            <Column header="Actions">
              <template #body="{ data }">
                <Button
                  icon="pi pi-users"
                  text
                  size="small"
                  v-tooltip="'Manage members'"
                  @click="openMembers(data)"
                />
                <Button
                  icon="pi pi-trash"
                  text
                  size="small"
                  severity="danger"
                  v-tooltip="'Delete group'"
                  @click="deleteGroup(data.id)"
                />
              </template>
            </Column>
          </DataTable>
        </div>

        <!-- Create Group Dialog -->
        <Dialog v-model:visible="showCreateGroup" header="Create Group" modal :style="{ width: '400px' }">
          <div class="flex flex-col gap-3">
            <div>
              <label class="block text-sm font-medium mb-1">Name</label>
              <InputText v-model="newGroup.name" class="w-full" placeholder="Group name" />
            </div>
            <div>
              <label class="block text-sm font-medium mb-1">Description</label>
              <InputText v-model="newGroup.description" class="w-full" placeholder="Optional description" />
            </div>
          </div>
          <template #footer>
            <Button label="Cancel" text @click="showCreateGroup = false" />
            <Button label="Create" :loading="saving" @click="createGroup" />
          </template>
        </Dialog>

        <!-- Manage Members Dialog -->
        <Dialog v-model:visible="showMembers" :header="`Members — ${activeGroup?.name}`" modal :style="{ width: '450px' }">
          <div class="flex gap-2 mb-3">
            <InputText v-model="newMemberEmail" class="flex-1" placeholder="Account email or ID" />
            <Button label="Add" size="small" :loading="addingMember" @click="addMember" />
          </div>
          <DataTable :value="groupMembers" empty-message="No members.">
            <Column field="account_id" header="Account ID" />
            <Column field="added_at" header="Added" />
            <Column header="">
              <template #body="{ data }">
                <Button icon="pi pi-times" text size="small" severity="danger" @click="removeMember(data.account_id)" />
              </template>
            </Column>
          </DataTable>
          <template #footer>
            <Button label="Close" text @click="showMembers = false" />
          </template>
        </Dialog>
      </TabPanel>

      <!-- Access Grants -->
      <TabPanel header="Access Grants">
        <div class="section">
          <div class="section-header">
            <h2>My Access Grants</h2>
            <Button label="New Grant" icon="pi pi-plus" size="small" @click="showCreateGrant = true" />
          </div>

          <DataTable :value="grants" :loading="loadingGrants" empty-message="No grants yet.">
            <Column field="granted_to" header="Target type" />
            <Column field="target_account_id" header="Account" />
            <Column field="target_group_id" header="Group" />
            <Column header="Permissions">
              <template #body="{ data }">
                <span v-if="data.can_view" class="mr-1 text-xs bg-green-100 text-green-800 px-1 rounded">view</span>
                <span v-if="data.can_comment" class="mr-1 text-xs bg-blue-100 text-blue-800 px-1 rounded">comment</span>
                <span v-if="data.can_label" class="mr-1 text-xs bg-yellow-100 text-yellow-800 px-1 rounded">label</span>
                <span v-if="data.can_download" class="mr-1 text-xs bg-purple-100 text-purple-800 px-1 rounded">download</span>
              </template>
            </Column>
            <Column field="granted_until" header="Expires" />
            <Column header="">
              <template #body="{ data }">
                <Button icon="pi pi-trash" text size="small" severity="danger" @click="revokeGrant(data.id)" />
              </template>
            </Column>
          </DataTable>
        </div>

        <!-- Create Grant Dialog -->
        <Dialog v-model:visible="showCreateGrant" header="Create Access Grant" modal :style="{ width: '480px' }">
          <div class="flex flex-col gap-3">
            <div>
              <label class="block text-sm font-medium mb-1">Grant to</label>
              <Select v-model="newGrant.granted_to" :options="grantedToOptions" option-label="label" option-value="value" class="w-full" />
            </div>
            <div v-if="newGrant.granted_to === 'account'">
              <label class="block text-sm font-medium mb-1">Account ID</label>
              <InputText v-model="newGrant.target_account_id" class="w-full" placeholder="Account UUID" />
            </div>
            <div v-if="newGrant.granted_to === 'group'">
              <label class="block text-sm font-medium mb-1">Group</label>
              <Select v-model="newGrant.target_group_id" :options="groups" option-label="name" option-value="id" class="w-full" />
            </div>
            <div class="flex flex-wrap gap-2">
              <label class="flex items-center gap-1 text-sm"><Checkbox v-model="newGrant.can_view" :binary="true" /> View</label>
              <label class="flex items-center gap-1 text-sm"><Checkbox v-model="newGrant.can_comment" :binary="true" /> Comment</label>
              <label class="flex items-center gap-1 text-sm"><Checkbox v-model="newGrant.can_label" :binary="true" /> Label</label>
              <label class="flex items-center gap-1 text-sm"><Checkbox v-model="newGrant.can_download" :binary="true" /> Download</label>
            </div>
            <div>
              <label class="block text-sm font-medium mb-1">Expires (optional)</label>
              <DatePicker v-model="newGrant.granted_until" class="w-full" show-time hour-format="24" />
            </div>
          </div>
          <template #footer>
            <Button label="Cancel" text @click="showCreateGrant = false" />
            <Button label="Create" :loading="saving" @click="createGrant" />
          </template>
        </Dialog>
      </TabPanel>

      <!-- Blocks -->
      <TabPanel header="Blocked Accounts">
        <div class="section">
          <div class="section-header">
            <h2>Blocked Accounts</h2>
            <Button label="Block Account" icon="pi pi-ban" size="small" severity="danger" @click="showBlockDialog = true" />
          </div>

          <DataTable :value="blocks" :loading="loadingBlocks" empty-message="No blocked accounts.">
            <Column field="blocked_id" header="Blocked Account ID" />
            <Column field="blocked_at" header="Blocked at" />
            <Column field="reason" header="Reason" />
            <Column header="">
              <template #body="{ data }">
                <Button label="Unblock" text size="small" @click="unblockAccount(data.blocked_id)" />
              </template>
            </Column>
          </DataTable>
        </div>

        <!-- Block Dialog -->
        <Dialog v-model:visible="showBlockDialog" header="Block Account" modal :style="{ width: '400px' }">
          <div class="flex flex-col gap-3">
            <div>
              <label class="block text-sm font-medium mb-1">Account ID</label>
              <InputText v-model="blockTarget.accountId" class="w-full" placeholder="Account UUID" />
            </div>
            <div>
              <label class="block text-sm font-medium mb-1">Reason (optional)</label>
              <InputText v-model="blockTarget.reason" class="w-full" placeholder="Reason" />
            </div>
          </div>
          <template #footer>
            <Button label="Cancel" text @click="showBlockDialog = false" />
            <Button label="Block" severity="danger" :loading="saving" @click="blockAccount" />
          </template>
        </Dialog>
      </TabPanel>
    </TabView>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import permissionService from '@/services/permissionService'
import TabView from 'primevue/tabview'
import TabPanel from 'primevue/tabpanel'
import DataTable from 'primevue/datatable'
import Column from 'primevue/column'
import Button from 'primevue/button'
import Dialog from 'primevue/dialog'
import InputText from 'primevue/inputtext'
import Select from 'primevue/select'
import Checkbox from 'primevue/checkbox'
import DatePicker from 'primevue/datepicker'

// ── Groups ────────────────────────────────────────────────────────────────────
const groups = ref([])
const loadingGroups = ref(false)
const showCreateGroup = ref(false)
const showMembers = ref(false)
const activeGroup = ref(null)
const groupMembers = ref([])
const newGroup = ref({ name: '', description: '' })
const newMemberEmail = ref('')
const addingMember = ref(false)
const saving = ref(false)

const loadGroups = async () => {
  loadingGroups.value = true
  try {
    const data = await permissionService.listGroups()
    groups.value = data?.groups ?? []
  } finally {
    loadingGroups.value = false
  }
}

const createGroup = async () => {
  saving.value = true
  try {
    await permissionService.createGroup(newGroup.value.name, newGroup.value.description)
    showCreateGroup.value = false
    newGroup.value = { name: '', description: '' }
    await loadGroups()
  } finally {
    saving.value = false
  }
}

const deleteGroup = async (groupId) => {
  await permissionService.deleteGroup(groupId)
  await loadGroups()
}

const openMembers = (group) => {
  activeGroup.value = group
  groupMembers.value = []
  showMembers.value = true
}

const addMember = async () => {
  if (!activeGroup.value || !newMemberEmail.value.trim()) return
  addingMember.value = true
  try {
    const result = await permissionService.addMember(activeGroup.value.id, newMemberEmail.value.trim())
    if (result?.success) {
      groupMembers.value = [...groupMembers.value, result.membership]
      newMemberEmail.value = ''
    }
  } finally {
    addingMember.value = false
  }
}

const removeMember = async (accountId) => {
  if (!activeGroup.value) return
  await permissionService.removeMember(activeGroup.value.id, accountId)
  groupMembers.value = groupMembers.value.filter(m => m.account_id !== accountId)
}

// ── Access Grants ─────────────────────────────────────────────────────────────
const grants = ref([])
const loadingGrants = ref(false)
const showCreateGrant = ref(false)
const grantedToOptions = [
  { label: 'Everyone (public)', value: 'everyone' },
  { label: 'Specific account', value: 'account' },
  { label: 'Group', value: 'group' },
]
const newGrant = ref({
  granted_to: 'everyone',
  target_account_id: '',
  target_group_id: null,
  can_view: true,
  can_comment: false,
  can_label: false,
  can_download: false,
  granted_until: null,
})

const loadGrants = async () => {
  loadingGrants.value = true
  try {
    const data = await permissionService.listGrants()
    grants.value = data?.grants ?? []
  } finally {
    loadingGrants.value = false
  }
}

const createGrant = async () => {
  saving.value = true
  try {
    const payload = { ...newGrant.value }
    if (payload.granted_until instanceof Date) {
      payload.granted_until = payload.granted_until.toISOString()
    } else if (payload.granted_until) {
      payload.granted_until = new Date(payload.granted_until).toISOString()
    }
    await permissionService.createGrant(payload)
    showCreateGrant.value = false
    newGrant.value = { granted_to: 'everyone', target_account_id: '', target_group_id: null, can_view: true, can_comment: false, can_label: false, can_download: false, granted_until: null }
    await loadGrants()
  } finally {
    saving.value = false
  }
}

const revokeGrant = async (grantId) => {
  await permissionService.revokeGrant(grantId)
  await loadGrants()
}

// ── Blocks ────────────────────────────────────────────────────────────────────
const blocks = ref([])
const loadingBlocks = ref(false)
const showBlockDialog = ref(false)
const blockTarget = ref({ accountId: '', reason: '' })

const loadBlocks = async () => {
  loadingBlocks.value = true
  try {
    const data = await permissionService.listBlocks()
    blocks.value = data?.blocks ?? []
  } finally {
    loadingBlocks.value = false
  }
}

const blockAccount = async () => {
  saving.value = true
  try {
    await permissionService.blockAccount(blockTarget.value.accountId, blockTarget.value.reason)
    showBlockDialog.value = false
    blockTarget.value = { accountId: '', reason: '' }
    await loadBlocks()
  } finally {
    saving.value = false
  }
}

const unblockAccount = async (accountId) => {
  await permissionService.unblockAccount(accountId)
  await loadBlocks()
}

onMounted(() => {
  loadGroups()
  loadGrants()
  loadBlocks()
})
</script>

<style scoped>
.permissions-container {
  padding: 1.5rem;
  max-width: 900px;
  margin: 0 auto;
}

h1 {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 1rem;
}

.section {
  padding: 0.5rem 0;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
}

h2 {
  font-size: 1rem;
  font-weight: 600;
}
</style>
