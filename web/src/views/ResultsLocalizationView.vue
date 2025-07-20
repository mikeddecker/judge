<script setup>
import { getColor } from '@/helpers/utils'
import { computed, onMounted, ref } from 'vue'

const props = defineProps({
  results: {
    type: Object,
    required: true,
  },
  frameLabelTypes: {
    type: Object,
    required: true
  }
})

const dailyChartData = ref(null)
const dailyChartDataCumulative = ref(null)

onMounted(async () => {
  dailyChartData.value = transformBoxCounts(props.results['boxcounts']['daily'], false)
  dailyChartDataCumulative.value = transformBoxCounts(props.results['boxcounts']['daily'], true)
})

const transformBoxCounts = (typedDays, cummulative) => {
  let datapoints = Object.fromEntries(Object.keys(props.frameLabelTypes).map(flt => [flt, []]))
  let indiviualOrCumulative = cummulative ? 'cumulative' : 'individual'

  let days = new Set()
  // Format values to an array per flt
  Object.entries(typedDays).forEach(([day, typedDay]) => {
    Object.entries(typedDay[indiviualOrCumulative]).forEach(([flt, count]) => {
      datapoints[flt].push(count)
      days.add(day)
    })
  })

  let datasets = Object.entries(datapoints).map(([flt, counts]) => {
    return {
      label: props.frameLabelTypes[flt],
      data: counts,
      borderColor: getColor(flt),
      fill: false,
      cubicInterpolationMode: 'monotone',
      tension: 0.4
    }
  })

  return {
    labels: Array.from(days),
    datasets: datasets
  };
}

const getChartOptions = (title) => {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: true,
        text: title
      },
    },
    interaction: {
      intersect: false,
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'day'
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'box count'
        },
        suggestedMin: 0,
        suggestedMax: 1,
        position: 'right'
      },
    }
  }
}

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
  <Chart v-if="dailyChartData" type="line" :data="dailyChartData" :options="getChartOptions('Daily box count')" class="h-[25rem]" />
  <Chart v-if="dailyChartDataCumulative" type="line" :data="dailyChartDataCumulative" :options="getChartOptions('Daily box count (cumulative)')" class="h-[25rem]" />

  <pre>{{ results }}</pre>
</template>

<style scoped>
</style>
