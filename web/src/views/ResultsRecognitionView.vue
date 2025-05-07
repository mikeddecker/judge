<script setup>
import { computed } from 'vue'

const props = defineProps({
  results: {
    type: Object,
    required: true,
  },
})

const selected = props.results['selected-model']
const totalAccuracy = props.results["modelcomparison"][selected]['accuracy']
const skillAccuracy = props.results["modelcomparison"][selected]['acc-skills']
const chartDataVal = computed(() => transformF1ToChart(props.results['f1-scores-val']))

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
      suggestedMax: 1
    }
  }
}

function getRandomInt(max) {
  return Math.floor(Math.random() * max);
}

const transformF1ToChart = (fscores) => {
  let DATA_COUNT = 12;
  let labels = [];
  for (let i = 0; i < DATA_COUNT; ++i) {
    labels.push(i.toString());
  }
  let key = 'Skill'
  let datapoints = {}
  
  console.log('val', Object.values(fscores))
  Object.values(fscores).forEach(
    (f, e) => {
      console.log('epoch', e)
      console.log('f', f)
      Object.entries(f).forEach(
        ( [skillprop, accuracy_value]) => {
          console.log(e, skillprop, accuracy_value)
          console.log('obj keys', skillprop, skillprop in datapoints, Object.keys(datapoints))
          if (skillprop in datapoints) {
            datapoints[skillprop].push(accuracy_value)
          } else {
            datapoints[skillprop] = [accuracy_value]
          }
        }
      )
    }
  )
  console.log(datapoints)
  let ds = Object.entries(datapoints).map(([k, v]) => {
    console.log(k, v)
    return {
      label: k,
      data: v,
      borderColor: `rgb(${getRandomInt(255)},${getRandomInt(255)},${getRandomInt(255)})`,
      fill: false,
      cubicInterpolationMode: 'monotone',
      tension: 0.4
    }
  })
  console.log(ds)
  // datapoints = [0.1, 0.20, 0.20, 0.60, 0.60, 0.120, NaN, 180, 120, 125, 105, 110, 170];
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
        <span class="text-2xl -pt-12">{{ totalAccuracy }}%</span>
      </template>
    </Card>
    
    <Card>
      <template #header>Skill Accuracy</template>
      <template #content>
        <span class="text-2xl -pt-12">{{ skillAccuracy }}%</span>
      </template>
    </Card>
  </div>

  <Chart type="line" :data="chartDataVal" :options="chartOptions" class="w-full" />
  <pre>{{ results }}</pre>
</template>

<style scoped>
.p-chart canvas {
  height: 30rem;
}

.p-card-body {
  padding-top: 0 !important;
}
</style>
