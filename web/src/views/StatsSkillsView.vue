<template>
  <pre>{{ Object.keys(results) }}</pre>

  <Tabs value="total">
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

const props = defineProps({
  results: {
    type: Object,
    required: true,
  },
})

onMounted(() => {
  console.log('Total prop_name_counts:', props.results['prop_name_counts']['total']);
  console.log('Total prop_value_frequencies:', props.results['prop_value_frequencies']['total']);

});

const transformCounts = (values) => {
    let labels = union(
        values['train'] ? Object.keys(values['train']) : [],
        values['test'] ? Object.keys(values['test']) : [],
    );
    let transformed = {
        labels,
        'datasets': [
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

<style scoped>

</style>

