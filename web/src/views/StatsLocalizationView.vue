<script setup>
import { useRouter } from 'vue-router'
import { formatPercentage, getColor } from '@/helpers/utils'
import { getDailyChartOptions, transformDailyCounts } from '@/helpers/chartUtils'
import { computed, onMounted, ref } from 'vue'
import SkillBalk from '@/components/SkillBalk.vue'

const router = useRouter()

const props = defineProps({
  results: {
    type: Object,
    required: true,
  },
  frameLabelTypes: {
    type: Object,
    required: true
  },
  videosWithLabels: {
    type: Array,
    required: true
  }
})

const dailyChartData = ref(null)
const dailyChartDataCumulative = ref(null)

onMounted(async () => {
  dailyChartData.value = transformDailyCounts(props.results['boxcounts']['daily'], props.frameLabelTypes, false)
  dailyChartDataCumulative.value = transformDailyCounts(props.results['boxcounts']['daily'], props.frameLabelTypes, true)
})

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
  <Chart v-if="dailyChartData" type="line" :data="dailyChartData" :options="getDailyChartOptions('Daily box count')" class="h-[25rem]" />
  <Chart v-if="dailyChartDataCumulative" type="line" :data="dailyChartDataCumulative" :options="getDailyChartOptions('Daily box count (cumulative)')" class="h-[25rem]" />

  <h3>Videos with Labels</h3>
  <DataTable
    :value="videosWithLabels" 
    scrollable scrollHeight="400px"
  >
    <Column
      v-for="(value, prop) in Object.values(videosWithLabels)[0]"
      :key="prop"
      sortable
      :field="prop"
      :header="prop"
    >
      <template #body="slotProps">
        {{ ['name'].includes(prop) ? slotProps.data[prop] : slotProps.data[prop] }}
      </template>
    </Column>
    <Column header="Link">
      <template #body="slotProps">
        <Button 
          @click="() => router.push(`/video/${slotProps.data.id}`)" 
          rounded icon="pi pi-external-link"
          variant="text"
          size="small" severity="secondary"></Button>
      </template>
    </Column>
    <Column header="Frame Labels">
      <template #body="slotProps">
        <SkillBalk 
          :videoinfo="{FrameLength: slotProps.data.frameLength}"
          :Skills="[]"
          :currentFrame="0"
          :labeledFrames="slotProps.data.labeledFrameNrs"
        />
      </template>
    </Column>
  </DataTable>

  <h3>Localization Recipes Performance</h3>
  <DataTable :value="Object.values(results['recipes'])">
    <Column
      v-for="(value, prop) in Object.values(results['recipes'])[0]"
      :key="prop"
      sortable
      :field="prop"
      :header="prop"
    >
      <template #body="slotProps">
        {{ ['model'].includes(prop) ? slotProps.data[prop] : formatPercentage(slotProps.data[prop]) }}
      </template>
    </Column>
  </DataTable>
</template>

<style scoped>
</style>

