<template>
    <DataTable
    :value="videosWithLabels" 
    scrollable scrollHeight="400px"
    sortField="density" :sortOrder="1"
  >
    <Column header="VideoId" sortable field="id"><template #body="slotProps">{{ slotProps.data.id }}</template></Column>
    <Column header="Name" sortable field="name"><template #body="slotProps">{{ slotProps.data.name }}</template></Column>
    <Column header="Distribution">
      <template #body="slotProps">
        <SkillBalk class="min-w-3xs"
          :videoinfo="{FrameLength: slotProps.data.frameLength}"
          :Skills="[]"
          :currentFrame="0"
          :labeledFrames="slotProps.data.labeledFrameNrs"
        />
      </template>
    </Column>
    <Column header="Density (Labels/sec)" sortable field="density"><template #body="slotProps">{{ slotProps.data.density.toFixed(2) }}</template></Column>
    <!-- <Column header="Labels x FPS / Duration" sortable field="labelsTimesFpsOverDuration"><template #body="slotProps">{{ slotProps.data.labelsTimesFpsOverDuration.toFixed(2) }}</template></Column> -->
    <Column header="Duration" sortable field="duration"><template #body="slotProps">{{ slotProps.data.duration }}</template></Column>
    <Column header="FPS" sortable field="fps"><template #body="slotProps">{{ slotProps.data.fps }}</template></Column>
    <Column header="Labeled Frames" sortable field="frameCount"><template #body="slotProps">{{ slotProps.data.frameCount }}</template></Column>
    <Column header="Total Frames" sortable field="frameLength"><template #body="slotProps">{{ slotProps.data.frameLength }}</template></Column>
    <Column header="Boxes" sortable field="totalBoxes"><template #body="slotProps">{{ slotProps.data.totalBoxes }}</template></Column>
    <Column header="Link">
      <template #body="slotProps">
        <Button 
          @click="() => router.push(`/video/${slotProps.data.id}`)" 
          rounded icon="pi pi-external-link"
          variant="text"
          size="small" severity="secondary"></Button>
      </template>
    </Column>
    <template #footer> In total there are {{ videosWithLabels ? videosWithLabels.length : 0 }} videos with location labels. </template>
  </DataTable>
</template>

<script setup>
import SkillBalk from '@/components/SkillBalk.vue'
import { useRouter } from 'vue-router'

const router = useRouter()

const props = defineProps({
    videosWithLabels: {
        type: Array,
        required: true
    }
})
</script>

