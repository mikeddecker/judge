<script setup>
import { formatPercentage, getColor, round2decimals } from '@/helpers/utils'
import { computed } from 'vue'

const props = defineProps({
  results: {
    type: Object,
    required: true,
  },
})

const selected = props.results['selected-model']
const f1MacroAvg = round2decimals(props.results['best']['f1_avg_accuracy'] * 100)
const f1MacroAvgSkills = round2decimals(props.results['best']['classification_reports'][props.results['best']['epoch']]['Skill']['macro avg']['f1-score'] * 100)
const chartDataVal = computed(() => transformF1ToChart(props.results['best']['f1_scores']))

const chartOptions = {
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
}


const transformF1ToChart = (fscores) => {
  let labels = Object.keys(fscores);
  
  let datapoints = {}
  Object.values(fscores).forEach(
    (f, e) => {
      Object.entries(f).forEach(
        ([skillprop, accuracy_value]) => {
          if (skillprop in datapoints) {
            datapoints[skillprop].push(accuracy_value)
          } else {
            datapoints[skillprop] = [accuracy_value]
          }
        }
      )
    }
  )
  let ds = Object.entries(datapoints).map(([k, v]) => {
    return {
      label: k,
      data: v,
      borderColor: getColor(k),
      fill: false,
      cubicInterpolationMode: 'monotone',
      tension: 0.4
    }
  })
  return {
    labels: labels,
    datasets: ds
  };
}

</script>


<template>
  <h2 class="mb-4">{{ selected }}</h2>
  <div class="flex gap-4">
    <Card>
      <template #header>Total Accuracy</template>
      <template #content>
        <span class="text-2xl">{{ f1MacroAvg }}%</span>
      </template>
    </Card>
    
    <Card>
      <template #header>Skill Accuracy</template>
      <template #content>
        <span class="text-2xl">{{ f1MacroAvgSkills }}%</span>
      </template>
    </Card>
  </div>

  <Chart type="line" :data="chartDataVal" :options="chartOptions" class="w-full" />

  <DataTable :value="Object.values(results['modelcomparison'])" class="w-2/3">
    <Column sortable field="model" header="model"></Column>
    
    <Column sortable field="f1-macro-avg" header="f1-macro-avg">
      <template #body="slotProps">
        {{ formatPercentage(slotProps.data['f1-macro-avg']) }}
      </template>
    </Column>

    <Column sortable field="f1-macro-avg-skills" header="f1-macro-avg-skills">
      <template #body="slotProps">
        {{ formatPercentage(slotProps.data['f1-macro-avg-skills']) }}
      </template>
    </Column>

    <Column sortable field="f1-weighted-avg" header="f1-weighted-avg">
      <template #body="slotProps">
        {{ formatPercentage(slotProps.data['f1-weighted-avg']) }}
      </template>
    </Column>

    <Column sortable field="f1-weighted-avg-skills" header="f1-weighted-avg-skills">
      <template #body="slotProps">
        {{ formatPercentage(slotProps.data['f1-weighted-avg-skills']) }}
      </template>
    </Column>

    <Column sortable field="total-accuracy" header="total-accuracy">
      <template #body="slotProps">
        {{ formatPercentage(slotProps.data['total-accuracy']) }}
      </template>
    </Column>
    <Column sortable field="date" header="date"></Column>
  </DataTable>
  <!-- <pre>{{ results }}</pre> -->
</template>

<style scoped>

</style>
