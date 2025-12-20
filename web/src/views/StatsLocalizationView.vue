<script setup>
import { useRouter } from 'vue-router'
import { formatPercentage, getColor } from '@/helpers/utils'
import { getDailyChartOptions, transformDailyCounts } from '@/helpers/chartUtils'
import { computed, onMounted, ref } from 'vue'
import SkillBalk from '@/components/SkillBalk.vue'

const router = useRouter()

const props = defineProps({
  results: {
    type: Object,
    required: true,
  },
  frameLabelTypes: {
    type: Object,
    required: true
  },
  videosWithLabels: {
    type: Array,
    required: true
  }
})

const dailyChartData = ref(null)
const dailyChartDataCumulative = ref(null)

onMounted(async () => {
  dailyChartData.value = transformDailyCounts(props.results['boxcounts']['daily'], props.frameLabelTypes, false)
  dailyChartDataCumulative.value = transformDailyCounts(props.results['boxcounts']['daily'], props.frameLabelTypes, true)
})

const barChartBoxesPerTypeData = computed(() => {
  return {
    'labels' : props.results['boxcounts']['total'].map(item => `${item.split}-${props.frameLabelTypes[item.type]}`),
    'datasets': [
      {
        label: 'Box count by Type',
        data: props.results['boxcounts']['total'].map(item => item.count),
        backgroundColor: [getColor(1), getColor(2)]
      }
    ]
  }
})

const barChartFramesTrainTest = computed(() => {
  return {
    'labels' : props.results['framecounts']['total'].map(item => `${item.split}`),
    'datasets': [
      {
        label: 'Frame count',
        data: props.results['framecounts']['total'].map(item => item.count),
        backgroundColor: [getColor(1), getColor(2)]
      }
    ]
  }
})
</script>

<template>
  <h2>Localization results</h2>
  <div class="flex justify-evenly">
    <Chart type="bar" :data="barChartFramesTrainTest" class="w-45/100"/>
    <Chart type="bar" :data="barChartBoxesPerTypeData" class="w-45/100"/>
  </div>
  <Chart v-if="dailyChartData" type="line" :data="dailyChartData" :options="getDailyChartOptions('Daily box count')" class="h-[25rem]" />
  <Chart v-if="dailyChartDataCumulative" type="line" :data="dailyChartDataCumulative" :options="getDailyChartOptions('Daily box count (cumulative)')" class="h-[25rem]" />

  <h3>Videos with Labels</h3>
  <DataTable
    :value="videosWithLabels" 
    scrollable scrollHeight="400px"
    sortField="density" sortOrder="1"
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

  <h3>Localization Recipes Performance</h3>
  <DataTable :value="Object.values(results['recipes'])">
    <Column
      v-for="(value, prop) in Object.values(results['recipes'])[0]"
      :key="prop"
      sortable
      :field="prop"
      :header="prop"
    >
      <template #body="slotProps">
        {{ ['model'].includes(prop) ? slotProps.data[prop] : formatPercentage(slotProps.data[prop]) }}
      </template>
    </Column>
  </DataTable>
</template>

<style scoped>
</style>

