<template>

  <DataTable
    :value="Object.values(ious)" 
    scrollable scrollHeight="400px"
    sortField="avg" :sortOrder="1"
  >
    <Column header="VideoId" sortable field="videoId"><template #body="slotProps">{{ slotProps.data.videoId }}</template></Column>
    <Column header="avg" sortable field="avg"><template #body="slotProps">{{ slotProps.data.avg.toFixed(2) }}</template></Column>
    <Column header="min" sortable field="avg"><template #body="slotProps">{{ slotProps.data.min.toFixed(2) }}</template></Column>
    <Column header="max" sortable field="max"><template #body="slotProps">{{ slotProps.data.max.toFixed(2) }}</template></Column>
    <!-- <Column header="min_second" sortable :field="slotProps.data.ious[1] ? slotProps.data.ious[1] : ''"><template #body="slotProps">{{ slotProps.data.avg.toFixed(2) }}</template></Column> -->
    <Column header="Link">
      <template #body="slotProps">
        <Button 
          @click="() => router.push(`/video/${slotProps.data.videoId}`)" 
          rounded icon="pi pi-external-link"
          variant="text"
          size="small" severity="secondary"></Button>
      </template>
    </Column>
    <template #footer> In total there are {{ ious ? ious.length : 0 }} validation videos with location labels. </template>
  </DataTable>

  <p></p>
  <pre>{{ ious }}</pre>
</template>

<script setup>
import SkillBalk from '@/components/SkillBalk.vue'
import { useRouter } from 'vue-router'
import { computed } from 'vue'

const router = useRouter()

const props = defineProps({
  ious: {
    type: Object,
    required: true
  }
})
</script>

