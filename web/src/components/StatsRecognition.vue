<template>
    <div class="flex flex-wrap gap-6 mb-16">
        <BarChartTrainTest
            :values="skillcounts"
            direction="x" title="Skill counts" class="flex-1"
        ></BarChartTrainTest>

        <BarChartTrainTest
            :values="skillCompositionCounts"
            direction="x" title="Composition counts" class="flex-2"
        ></BarChartTrainTest>
    </div>

    <Chart v-if="dailyChartData" type="line" :data="dailyChartData" :options="getDailyChartOptions('Daily skill count', 'skills')" class="h-[25rem]" />
    <Chart v-if="dailyChartDataCumulative" type="line" :data="dailyChartDataCumulative" :options="getDailyChartOptions('Daily skill count (cumulative)', 'skills', dailyChartDataCumulativeHlines)" class="h-[25rem]" />

    <pre>{{ stats['layercomposition_names'] }}</pre>

    <p>
      TODO: layer_counts : DDSwitch : {} Jumper : {} SingleRope : {} Turner : {}
    </p>

    <h3>Layer value frequencies</h3>
    <Tabs value="total" class="mt-8">
      <TabList>
        <Tab
          v-for="layercomposition in Object.keys(stats['prop_value_frequencies'])"
          :key="layercomposition"
          :value="layercomposition"
          >
          {{ layercomposition }}
        </Tab>
      </TabList>

      <TabPanels>
        <TabPanel
          v-for="layercomposition in Object.keys(stats['prop_value_frequencies'])"
          :key="layercomposition"
          :value="layercomposition"
        >
          <!-- Different kind of stats, other method required & individual vs total-->
          <!-- <BarChartTrainTest
            :values="transformPropValueFrequencyCounts(stats['layer_counts'][layercomposition])"
            direction="y"
            :squared="true"
            :title="`Layer counts ${layercomposition}`"
          /> -->
          <div class="flex flex-wrap gap-8">
            <BarChartTrainTest
              v-for="(values, layer) in stats['prop_value_frequencies'][layercomposition]"
                :values="transformPropValueFrequencyCounts(values)"
                direction="x"
                :title="layer"
                class="w-120 flex-auto"
                :squared="true"
            />
          </div>
        </TabPanel>
      </TabPanels>
    </Tabs>

</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import BarChartTrainTest from './BarChartTrainTest.vue';
import { formatPercentage, getColor, round2decimals, union } from '@/helpers/utils';
import { getDailyChartOptions, transformDailyCounts } from '@/helpers/chartUtils';

const props = defineProps({
    stats: {
        type: Object,
        required: true,
    }
})

const dailyChartData = ref(null)
const dailyChartDataCumulative = ref(null)
const dailyChartDataCumulativeHlines = ref([])
// TODO: re-add/think about re-adding hlines of train/test size in graph
// [results['models'][selectedModel]['length_train'], results['models'][selectedModel]['length_val']]

const skillcounts = computed(() => {
  return {
    labels: ['count'],
    datasets: [
      {
        backgroundColor: getColor(1),
        data: [props.stats['recognition_counts']['total']['train']],
        label: 'Train'
      },
      {
        backgroundColor: getColor(2),
        data: [props.stats['recognition_counts']['total']['test']],
        label: 'Test'
      },
    ]
  }
})

const skillCompositionCounts = computed(() => {
  return {
    labels: Object.keys(props.stats['layercomposition_counts']),
    datasets: [
      {
        backgroundColor: getColor(1),
        data: Object.values(props.stats['layercomposition_counts']).map(train_test_values => train_test_values['train']),
        label: 'Train'
      },
      {
        backgroundColor: getColor(2),
        data: Object.values(props.stats['layercomposition_counts']).map(train_test_values => train_test_values['test']),
        label: 'Test'
      },
    ]
  }
})

onMounted(() => {
  dailyChartData.value = transformDailyCounts(props.stats['recognition_counts']['daily'], { 'train': 'train', 'test': 'test' }, false)
  dailyChartDataCumulative.value = transformDailyCounts(props.stats['recognition_counts']['daily'], { 'train': 'train', 'test': 'test' }, true)
});

const transformPropValueFrequencyCounts = (values) => {
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
</script>

