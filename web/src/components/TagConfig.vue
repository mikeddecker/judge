<template>
  <DataTable v-model:expandedRowGroups="expandedRowGroups" :value="tags" rowGroupMode="subheader" groupRowsBy="TagGroup.Name" expandableRowGroups sortMode="single" 
    sortField="TagGroup.Name" :sortOrder="1" scrollable scrollHeight="600px" editMode="cell">
    <Column field="Name" header="Name">
      <template #editor="{ data, field }">
        <InputText v-model="data.Name" @update:model-value="(newName) => onTagNameChanged(data.Id, newName)"></InputText>
      </template>
    </Column>
    <Column field="Keywords" header="Keywords (comma separated)">
      <template #editor="{ data, field }">
        <InputText v-model="data.Keywords" @update:model-value="(newValue) => onTagKeywordsChanged(data.Id, newValue)"></InputText>
      </template>
    </Column>
    <Column field="TagGroup" header="TagGroup">
      <template #body="{ data, field }">
        {{ data.TagGroup.Name }}
      </template>
      <template #editor="{ data, field }">
        <Select v-model="data.TagGroup.Name" :options="tagGroups" @update:model-value="(newGroup) => onTagGroupChanged(data.Id, newGroup)"></Select>
      </template>
    </Column>
    
    <template #groupheader="slotProps">
      <img :alt="slotProps.data.TagGroup.Name" src="@/assets/geometry.png" width="32" style="vertical-align: middle; display: inline-block" class="ml-2" />
      <span class="align-middle ml-2 font-bold leading-normal">{{ slotProps.data.TagGroup.Name }}</span>
    </template>
    

    <template #footer>
      <div class="flex gap-2 mb-2">
        <IftaLabel>
          <InputText id="new_tag" v-model="new_tag" variant="filled" :disabled="!new_tag_group_is_empty && new_tag_is_empty" @update:model-value="handleNewNameFooter"/>
          <label for="new_tag">New tag</label>
        </IftaLabel>
        
        <IftaLabel v-if="new_tag_is_empty">
          <InputText id="new_tag_group" v-model="new_tag_group" variant="filled" :disabled="!new_tag_is_empty" />
          <label for="new_tag_group">New tag group</label>
        </IftaLabel>
        <Select v-else v-model="new_tag_group" :options="tagGroups"></Select>

        <Button v-if="!new_tag_is_empty" class="" icon="pi pi-database" @click="() => addTag(new_tag, new_tag_group).then(() => refreshTags())" label="Add tag" aria-label="Add tag"></Button>
        <Button v-if="new_tag_is_empty && !new_tag_group_is_empty" class="" icon="pi pi-database" @click="() => addTagGroup(new_tag_group).then(() => refreshTags())" label="Add tag group" aria-label="Add tag group"></Button>
      </div>
      Total {{ tags ? tags.length : 0 }} tags in {{ tagGroups ? tagGroups.length : 0 }} groups. {{ tagGroups }}
    </template>
  </DataTable>
</template>

<script setup>
import { isNullOrWhiteSpace } from '@/helpers/utils';
import { addTag, addTagGroup, getTagGroups, getTags, updateTag, updateTagGroup } from '@/services/videoService';
import { InputText } from 'primevue';
import { computed, onMounted, ref } from 'vue';

const tags = ref(null)
const tagGroups = ref(null)
const new_tag = ref('')
const new_tag_group = ref('')
const new_tag_is_empty = computed(() => isNullOrWhiteSpace(new_tag.value))
const new_tag_group_is_empty = computed(() => isNullOrWhiteSpace(new_tag_group.value))
const expandedRowGroups = ref([])

onMounted(async () => {
  refreshTags()
})

const refreshTags = async () => {
  getTags().then(t => tags.value = t.map(tag => ({...tag,  TagGroup: tag.TagGroup || { Name: 'Ungrouped' }})))
  getTagGroups().then(tgs => {
    let names = tgs.map(t => t.Name)
    names.push('Ungrouped')
    tagGroups.value = names
  })
}

const onTagGroupChanged = (id, newGroup) => {
  updateTagGroup(id, newGroup == 'Ungrouped' ? undefined : newGroup)
}

const onTagNameChanged = (id, newName) => {
  updateTag(id, newName, null)
  tags.value.filter(t => t.Id == id)[0].Name = newName
}

const onTagKeywordsChanged = (id, newValue) => {
  updateTag(id, null, newValue)
  tags.value.filter(t => t.Id == id)[0].Keywords = newValue
}
const handleNewNameFooter = (newName) => {
  if (isNullOrWhiteSpace(newName)) { new_tag_group.value = null }
}

</script>