<script setup>
import FolderInfo from './FolderInfo.vue';

const props = defineProps({
  title: {
    required: false,
    type: String,
  },
  folders: {
    required: false,
    type: Array,
  },
  parentId: {
    required: false,
    type: String
  }
})
</script>

<template>
  <div class="container flex flex-wrap">
    <FolderInfo
      :folder-id="parentId" title="go back"
      @changeFolder="$emit('changeFolder', parentId)">
    </FolderInfo>
    <FolderInfo v-for="folder in folders"
      :key="folder.Id" :folder-id="folder.Id" :title="folder.Name"
      :folderinfo="folder"
      @changeFolder="$emit('changeFolder', folder.Id)">
    </FolderInfo>
  </div>
</template>

<style scoped>
.container {
  justify-content: left;
}

@media (min-width: 1024px) {
}
</style>

