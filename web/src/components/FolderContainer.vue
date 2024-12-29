<script setup>
import FolderInfo from './FolderInfo.vue';

defineProps(['title', 'folders'])
</script>

<template>
  <div class="container">
    <FolderInfo v-for="folder in folders" :key="folder.Id" :folder-id="folder.Id" :title="folder.Name"/>
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
