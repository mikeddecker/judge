<template>
  <StatsLocalization v-if="stats" :stats="stats"></StatsLocalization>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import StatsLocalization from '@/components/StatsLocalization.vue'
import { getStats } from '@/services/videoService'

const stats = ref(undefined)

onMounted(async () => {
  getStats('localization').then(r => {
    stats.value = r
  })
})
</script>

<style scoped>
</style>

