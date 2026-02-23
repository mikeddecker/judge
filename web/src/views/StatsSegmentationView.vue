<template>
    <h1>Stats</h1>
    <Message v-if="!displayStats" severity="warn" variant="outlined" icon="pi pi-microchip-ai">
        Label a skill/segment first to see segment statistics
    </Message>

    <pre v-if="displayStats">{{ stats }}</pre>
</template>

<script setup>
import { getSkillCount, getStats } from '@/services/videoService';
import { ref, onMounted } from 'vue'

const displayStats = ref(false)
const stats = ref(undefined)

onMounted(async () => {
    getSkillCount().then(r => displayStats.value = r > 0 ? true : false)
    getStats('segmentation').then(r => stats.value = r)
})
</script>

