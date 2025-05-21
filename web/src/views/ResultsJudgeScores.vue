<script setup>
import { computed } from 'vue'

const props = defineProps({
  results: {
    type: Object,
    required: true,
  },
})

const total = computed(() => props.results["total"])
const resultsWithoutTotal = computed(() => {
  let clone = {...props.results}
  delete clone.total
  return Object.entries(clone).map(s => s[1])
})

</script>


<template>
  <h2 class="mb-4">Judge scores</h2>
  <DataTable :value="resultsWithoutTotal" class="w-1/2">
    <Column field="videoId" header="videoId"></Column>
    <Column field="judges" header="judges"></Column>
    <Column field="HAR_MViT" header="HAR_MViT"></Column>
    <Column field="HAR_MViT_difference" header="Difference"></Column>
  </DataTable>

  <pre>Total: {{ total }}</pre>
</template>

<style scoped>

</style>
