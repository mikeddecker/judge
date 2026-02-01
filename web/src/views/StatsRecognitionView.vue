<template>
    <h2>Stats</h2>
    <StatsRecognition v-if="stats" :stats="stats"/>
</template>

<script setup>
import StatsRecognition from '@/components/StatsRecognition.vue';
import { getStats } from '@/services/videoService';
import { ref, onMounted } from 'vue'

const stats = ref(undefined)

onMounted(async () => {
    getStats('recognition').then(r => stats.value = r).then(() => console.log('recognition stats', stats.value))
})
</script>

