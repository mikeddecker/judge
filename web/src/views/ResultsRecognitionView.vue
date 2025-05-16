<script setup>
import { computed } from 'vue'

const props = defineProps({
  results: {
    type: Object,
    required: true,
  },
})

const selected = props.results['selected-model']
const totalAccuracy = round2decimals(props.results["modelcomparison"][selected]['accuracy'] * 100)
const skillAccuracy = round2decimals(props.results["modelcomparison"][selected]['acc-skills'] * 100)
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
      suggestedMax: 1,
      position: 'right'
    },
  }
}

function getRandomInt(max) {
  return Math.floor(Math.random() * max);
}

function round2decimals(i) {
  return Math.round(i * 100) / 100
}

function getColor(skillprop) {
  switch (skillprop) {
    case 'Total':
      return `rgb(150, 50, 0)`
    case 'Skill':
      return `rgb(0, 20, 20)`
    default:
      let greencolor = 80 + getRandomInt(175)
      return `rgb(${greencolor * Math.random()}, ${190 + getRandomInt(65)}, ${greencolor})`
      return `rgb(${getRandomInt(255)}, ${getRandomInt(255)}, ${getRandomInt(255)})`
  }
}

const transformF1ToChart = (fscores) => {
  let labels = Object.keys(fscores);
  console.log("f1-scores", )
  console.log("labels", labels)
  let key = 'Skill'
  
  let datapoints = {}
  Object.values(fscores).forEach(
    (f, e) => {
      Object.entries(f).forEach(
        ( [skillprop, accuracy_value]) => {
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
  <!-- <pre>{{ results }}</pre> -->
</template>

<style scoped>
.p-chart canvas {
  height: 30rem;
}

.p-card-body {
  padding-top: 0 !important;
}
</style>
