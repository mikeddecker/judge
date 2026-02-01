<template>
    <h1>Stats</h1>
    <!-- <pre>{{ stats }}</pre> -->

    <div class="flex justify-evenly" v-if="frameLabelTypes">
      <Chart type="bar" :data="barChartFramesTrainTest" class="w-45/100"/>
      <Chart type="bar" :data="barChartBoxesPerTypeData" class="w-45/100"/>
    </div>
    <Chart v-if="dailyChartData" type="line" :data="dailyChartData" :options="getDailyChartOptions('Daily box count')" class="h-[25rem]" />
    <Chart v-if="dailyChartDataCumulative" type="line" :data="dailyChartDataCumulative" :options="getDailyChartOptions('Daily box count (cumulative)')" class="h-[25rem]" />

    <h2>Videos with labels</h2>
    <TableDensity :videos-with-labels="stats['labelinfo_per_video']"></TableDensity>

</template>

<script setup>

import { getColor } from '@/helpers/utils'
import { getFrameLabelTypes } from '@/services/videoService'
import { computed, onMounted, ref } from 'vue'
import TableDensity from './TableDensity.vue'
import { getDailyChartOptions, transformDailyCounts } from '@/helpers/chartUtils'

const props = defineProps({
  stats: {
    type: Object,
    required: true,
  }
})

const barChartBoxesPerTypeData = computed(() => {
  return {
    'labels' : props.stats['boxcounts']['total'].map(item => {
      return `${item.split}-${frameLabelTypes.value[item.type]}`
    }),
    'datasets': [
      {
        label: 'count',
        data: props.stats['boxcounts']['total'].map(item => item.count),
        backgroundColor: [getColor(1), getColor(2)]
      }
    ]
  }
})

const barChartFramesTrainTest = computed(() => {
  return {
    'labels' : props.stats['framecounts']['total'].map(item => `${item.split}`),
    'datasets': [
      {
        label: 'Frame count',
        data: props.stats['framecounts']['total'].map(item => item.count),
        backgroundColor: [getColor(1), getColor(2)]
      }
    ]
  }
})

const dailyChartData = ref(null)
const dailyChartDataCumulative = ref(null)
const frameLabelTypes = ref(undefined)

onMounted(async () => {
  getFrameLabelTypes().then(
    types => frameLabelTypes.value = types
  ).then(
    () => {
      dailyChartData.value = transformDailyCounts(props.stats['boxcounts']['daily'], frameLabelTypes.value, false)
      dailyChartDataCumulative.value = transformDailyCounts(props.stats['boxcounts']['daily'], frameLabelTypes.value, true)
    }
  )
})
</script>

