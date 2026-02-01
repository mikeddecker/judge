<template>
  <h1>{{ results['models'][selectedModel]['modelname'] }} - {{ results['models'][selectedModel]['rundate'] }}</h1>
  <span>f1 average over time: {{ formatPercentage(f1_avg) }}</span>
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


</script>

<style scoped>

</style>

