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
    'HAR_MViT',
    'HAR_Resnet_MC3',
    // 'HAR_SA_Conv3D',
    'HAR_Resnet_R2plus1',
    'HAR_SwinT_t',
    'HAR_SwinT_s',
    'HAR_Resnet_R3D',
    'HAR_MViT_extra_dense',
]

</script>


<template>
  <h2 class="mb-4">Judge scores</h2>
  <DataTable :value="resultsWithoutTotal">
    <Column sortable field="videoId" header="videoId"></Column>
    <Column sortable field="judges" header="judges"></Column>
    <Column sortable v-for="m in models" :field="m" :header="m"></Column>
    <Column sortable v-for="m in models" :field="m + '_procent_difference'" :header="m + '% diff'"></Column>
  </DataTable>

  <pre>Total: {{ total }}</pre>
</template>

<style scoped>

</style>
