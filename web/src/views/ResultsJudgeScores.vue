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

const models = [
    'MViT',
    // 'Resnet_MC3',
    // 'SA_Conv3D',
    // 'Resnet_R2plus1',
    'SwinT_t',
    'SwinT_s',
    'Resnet_R3D',
    'MViT_extra_dense',
]

</script>


<template>
  <h2 class="mb-4">Judge scores</h2>
  <DataTable :value="resultsWithoutTotal">
    <Column sortable field="videoId" header="videoId"></Column>
    <Column sortable field="judges" header="judges"></Column>
    <Column sortable v-for="m in models" :field="m" :header="m"></Column>
    <Column sortable v-for="m in models" :field="m + '_procent_difference'" :header="m + ' % diff'"></Column>
  </DataTable>

  <pre>Total: {{ total }}</pre>
</template>

<style scoped>

</style>
