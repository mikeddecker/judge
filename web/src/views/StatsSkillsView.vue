<template>
  <h1>{{ results['models'][selectedModel]['modelname'] }} - {{ results['models'][selectedModel]['rundate'] }}</h1>
  <span>f1 average over time: {{ formatPercentage(f1_avg) }}</span>

  <Chart type="line" :data="chartDataBestModel" :options="chartOptionsBestModel" />
</template>

<script setup>
import { formatPercentage, getColor, round2decimals, union } from '@/helpers/utils'
import { computed, nextTick, onMounted, ref } from 'vue'
import BarChartTrainTest from '@/components/BarChartTrainTest.vue'
import ConfusionMatrix from '@/components/ConfusionMatrix.vue';
import { getDailyChartOptions, transformDailyCounts } from '@/helpers/chartUtils';

const props = defineProps({
  results: {
    type: Object,
    required: true,
  },
})

const selectedModel = ref('best')

const f1_avg = computed(() => {
  let f1_avg_over_time = props.results.models[selectedModel.value]['f1_total_avgs_over_time']
  return f1_avg_over_time[f1_avg_over_time.length - 1]
})

const chartDataBestModel = computed(() => {
  let metricsOverTime = props.results['models'][selectedModel.value]['metrics_over_time']
  const classes = Object.keys(metricsOverTime[0]['f1'])
  const epochs = Object.entries(metricsOverTime).map((_, idx) => `${idx}`)

  // Create one dataset per class
  const datasets = classes.map(cls => ({
    label: cls,
    data: Object.values(metricsOverTime).map(entry => entry['f1'][cls]),
    borderColor: getColor(cls),
    fill: false,
    cubicInterpolationMode: 'monotone',
    tension: 0.4,
  }))

  return {
    labels: epochs,
    datasets
  }
})

const chartOptionsBestModel = ref({
  responsive: true,
  plugins: {
    title: {
      display: true,
      text: 'F1-scores-validation'
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
        text: 'epoch'
      }
    },
    y: {
      display: true,
      title: {
        display: true,
        text: 'f1-score'
      },
      suggestedMin: 0,
      suggestedMax: 1,
      position: 'right'
    },
  }
})

</script>

<style scoped>

</style>

