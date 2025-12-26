<script setup>
import { useRouter } from 'vue-router'
import { formatPercentage, getColor } from '@/helpers/utils'
import { getDailyChartOptions, transformDailyCounts } from '@/helpers/chartUtils'
import { computed, onMounted, ref } from 'vue'
import TableDensity from '@/components/TableDensity.vue'
import TableIous from '@/components/TableIous.vue'

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

const models = computed(() => Object.keys(props.results['recipes']))
const smoothTechniques = computed(() => selectedModel.value ? Object.keys(props.results['recipes'][selectedModel.value]['ious']) : [])
const selectedModel = ref(Object.keys(props.results['recipes'])[0])
const selectedSmoothTechnique = ref(Object.keys(props.results['recipes'][selectedModel.value]['ious'])[0])
const tabledIous = computed(() => {
  return Object.fromEntries(
    Object.entries(props.results['recipes'][selectedModel.value]['ious'][selectedSmoothTechnique.value]['val']['videos']).map(
      ([videoId, scores]) => [videoId, {
        videoId: Number(videoId),
        ...scores,
        labels: scores.ious.length
      }]
    )
  )
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
  <TableDensity :videos-with-labels="videosWithLabels"></TableDensity>

  <h3>Localization Recipes Performance</h3>
  <DataTable :value="Object.values(results['recipes'])">
    <Column key="model" field="model" header="model" sortable></Column>
    <Column key="team_raw_avg" field="team_raw_avg" header="team_raw_avg" sortable><template #body="slotProps">{{ formatPercentage(slotProps.data['team_raw_avg']) }}</template></Column>
    <Column key="team_smoothing_avg" field="team_smoothing_avg" header="team_smoothing_avg" sortable><template #body="slotProps">{{ formatPercentage(slotProps.data['team_smoothing_avg']) }}</template></Column>
    <Column key="fitness" field="fitness" header="fitness" sortable><template #body="slotProps">{{ formatPercentage(slotProps.data['fitness']) }}</template></Column>
    <Column key="metrics/mAP50(B)" field="metrics/mAP50(B)" header="metrics/mAP50(B)" sortable><template #body="slotProps">{{ formatPercentage(slotProps.data['metrics/mAP50(B)']) }}</template></Column>
    <Column key="metrics/mAP50-95(B)" field="metrics/mAP50-95(B)" header="metrics/mAP50-95(B)" sortable><template #body="slotProps">{{ formatPercentage(slotProps.data['metrics/mAP50-95(B)']) }}</template></Column>
    <Column key="metrics/precision(B)" field="metrics/precision(B)" header="metrics/precision(B)" sortable><template #body="slotProps">{{ formatPercentage(slotProps.data['metrics/precision(B)']) }}</template></Column>
    <Column key="metrics/recall(B)" field="metrics/recall(B)" header="metrics/recall(B)" sortable><template #body="slotProps">{{ formatPercentage(slotProps.data['metrics/recall(B)']) }}</template></Column>
  </DataTable>

  <h3>Performance per video</h3>
  <div class="flex gap-2 my-4">
    <Button 
      v-for="model in models" :label="model" :aria-label="model"
      severity="success" variant="text" raised
      :class="selectedModel == model ? 'p-button-highlight' : ''"
      @click="selectedModel = model"
    ></Button>
    <Button 
      v-for="smoothTechnique in smoothTechniques" :label="smoothTechnique" :aria-label="smoothTechnique"
      severity="success" variant="text" raised
      :class="selectedSmoothTechnique == smoothTechnique ? 'p-button-highlight' : ''"
      @click="selectedSmoothTechnique = smoothTechnique"
    ></Button>
  </div>

  <TableIous :ious="tabledIous"></TableIous>
</template>

<style scoped>
</style>

