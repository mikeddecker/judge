<template>
    <h1>Stats</h1>
    <Message v-if="!displayStats" severity="warn" variant="outlined" icon="pi pi-microchip-ai">
        Label a skill first to see skill statistics
    </Message>
    <StatsRecognition v-if="stats && displayStats" :stats="stats"/>
</template>

<script setup>
import StatsRecognition from '@/components/StatsRecognition.vue';
import { getSkillCount, getStats } from '@/services/videoService';
import { ref, onMounted } from 'vue'

const displayStats = ref(false)

const stats = ref(undefined)

onMounted(async () => {
    getSkillCount().then(r => displayStats.value = r > 0 ? true : false)
    getStats('recognition').then(r => stats.value = r).then(() => console.log('recognition stats', stats.value))
})
</script>

