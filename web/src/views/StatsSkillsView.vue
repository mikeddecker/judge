<template>
  <div class="flex flex-wrap gap-6 mb-16">
    <BarChartTrainTest :values="skillcounts" 
    direction="x" title="Skill counts" class="flex-1"
    ></BarChartTrainTest>
    
    <BarChartTrainTest :values="skillCompositionCounts" 
    direction="x" title="composition counts" class="flex-2"
    ></BarChartTrainTest>
  </div>
  
  <Chart v-if="dailyChartData" type="line" :data="dailyChartData" :options="getDailyChartOptions('Daily skill count', 'skills')" class="h-[25rem]" />
  <Chart v-if="dailyChartDataCumulative" type="line" :data="dailyChartDataCumulative" :options="getDailyChartOptions('Daily skill count (cumulative)', 'skills')" class="h-[25rem]" />
  
  <Chart type="line" :data="chartDataBestModel" :options="chartOptionsBestModel" />
  
  <ConfusionMatrix v-for="(matrix, prop) in results['models'][selectedModel]['validation_results']['metrics']['confusion']" :name="prop" :confusion="matrix"></ConfusionMatrix>
  
  <Tabs value="total" class="mt-8">
    <TabList>
      <Tab value="total">Total</Tab>
      <Tab
      v-for="layercomposition in results['layercomposition_names']"
      :key="layercomposition"
      :value="layercomposition"
      >
      {{ layercomposition }}
    </Tab>
  </TabList>
  
  <TabPanels>
    <TabPanel value="total">
      <BarChartTrainTest
      :values="transformCounts(results['prop_name_counts']['total'])"
      direction="y"
      title="Total property counts"
      :squared="true"
      />
      <div class="flex flex-wrap gap-8">
        <BarChartTrainTest
        v-for="(values, property) in results['prop_value_frequencies']['total']"
        :values="transformCounts(values)"
        direction="x"
        :title="property"
        class="w-120 flex-auto"
        :squared="true"
        />
        </div>
      </TabPanel>

      <TabPanel
      v-for="layercomposition in results['layercomposition_names']"
      :key="layercomposition"
      :value="layercomposition"
      >
      <BarChartTrainTest
      :values="transformCounts(results['prop_name_counts'][layercomposition])"
      direction="y"
      :squared="true"
      :title="`Property counts ${layercomposition}`"
      />
      <div class="flex flex-wrap gap-8">
        <BarChartTrainTest
        v-for="(values, property) in results['prop_value_frequencies'][layercomposition]"
        :values="transformCounts(values)"
        direction="x"
        :title="property"
        class="w-120 flex-auto"
        :squared="true"
        />
      </div>
    </TabPanel>
  </TabPanels>
</Tabs>

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

const dailyChartData = ref(null)
const dailyChartDataCumulative = ref(null)

onMounted(() => {
  dailyChartData.value = transformDailyCounts(props.results['skills']['daily'], { 'train': 'train', 'test': 'test' }, false)
  dailyChartDataCumulative.value = transformDailyCounts(props.results['skills']['daily'], { 'train': 'train', 'test': 'test' }, true)
});

const transformCounts = (values) => {
    let labels = union(
        values['train'] ? Object.keys(values['train']) : [],
        values['test'] ? Object.keys(values['test']) : [],
    );
    let transformed = {
        labels,
        datasets: [
            {
                label: 'Train',
                data: labels.map(label => Math.sqrt(values['train']?.[label]) || 0),
                backgroundColor: getColor(1),
            },
            {
                label: 'Test',
                data: labels.map(label => Math.sqrt(values['test']?.[label]) || 0),
                backgroundColor: getColor(2),
            },
        ]
    }
    return transformed
}

const skillcounts = computed(() => { 
  return {
    labels: ['count'],
    datasets: [
      { 
        backgroundColor: getColor(1),
        data: [props.results['skills']['total']['train']],
        label: 'Train'
      },
      { 
        backgroundColor: getColor(2),
        data: [props.results['skills']['total']['test']],
        label: 'Test'
      },
    ]
  }
})

const skillCompositionCounts = computed(() => {
  return {
    labels: Object.keys(props.results['layercomposition_counts']),
    datasets: [
      { 
        backgroundColor: getColor(1),
        data: Object.values(props.results['layercomposition_counts']).map(train_test_values => train_test_values['train']),
        label: 'Train'
      },
      { 
        backgroundColor: getColor(2),
        data: Object.values(props.results['layercomposition_counts']).map(train_test_values => train_test_values['test']),
        label: 'Test'
      },
    ]
  }
})

const chartDataBestModel = computed(() => {
  console.log(props.results)
  let metricsOverTime = props.results['models'][selectedModel.value]['metrics_over_time']
  console.log('metrics over time', metricsOverTime)
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

