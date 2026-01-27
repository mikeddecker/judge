<template>
    <h1>Stats</h1>
    <pre>{{ stats }}</pre>
</template>

<script setup>
import { getStats } from '@/services/videoService';
import { ref, onMounted } from 'vue'

const stats = ref(undefined)

onMounted(async () => {
    getStats('recognition').then(r => stats.value = r)
})
</script>

