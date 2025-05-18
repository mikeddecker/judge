<script setup>
import { computed } from 'vue'

const props = defineProps({
  results: {
    type: Object,
    required: true,
  },
})

</script>


<template>
  <h2 class="mb-4">Judge scores</h2>
  <DataTable :value="results" class="w-1/2">
    <Column field="videoId" header="videoId"></Column>
    <Column field="judges" header="judges"></Column>
    <Column field="HAR_MViT" header="HAR_MViT"></Column>
    <Column field="HAR_MViT_difference" header="Difference"></Column>
  </DataTable>
</template>

<style scoped>

</style>
