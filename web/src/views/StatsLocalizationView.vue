<template>

  <pre>{{ selectedRecipe }}</pre>
  <StatsLocalization v-if="stats" :stats="stats"></StatsLocalization>
  <pre>{{ frameLabelTypes }}</pre>
  <pre>{{ stats }}</pre>
</template>

<script setup>
import { useRouter } from 'vue-router'
import { formatPercentage, getColor } from '@/helpers/utils'
import { getDailyChartOptions, transformDailyCounts } from '@/helpers/chartUtils'
import { computed, onMounted, ref } from 'vue'
import TableDensity from '@/components/TableDensity.vue'
import TableIous from '@/components/TableIous.vue'
import StatsLocalization from '@/components/StatsLocalization.vue'
import { getFrameLabelTypes, getStats } from '@/services/videoService'

const router = useRouter()
const stats = ref(undefined)
const selectedRecipe = ref(undefined)
const frameLabelTypes = ref(undefined)

onMounted(async () => {
  getStats('localization').then(r => {
    stats.value = r
  })
})
</script>

<style scoped>
</style>

