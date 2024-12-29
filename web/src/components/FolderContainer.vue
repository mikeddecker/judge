<script setup>
import FolderInfo from './FolderInfo.vue';

defineProps(['title', 'folders', 'parentId'])

</script>

<template>
  <div class="container">
    <FolderInfo :folder-id="parentId" title="go back" @changeFolder="$emit('changeFolder', parentId)"/>
    <FolderInfo v-for="folder in folders" :key="folder.Id" :folder-id="folder.Id" :title="folder.Name" @changeFolder="$emit('changeFolder', folder.Id)"/>
  </div>
</template>

<style scoped>
.container {
  display: flex;
  justify-content: left;
  flex-wrap: wrap;
}

@media (min-width: 1024px) {
}
</style>
