<template>
    <h1>Results</h1>
    <pre>{{ results }}</pre>
</template>

<script setup>

import { getResults } from '@/services/videoService';
import { ref, onMounted } from 'vue'

// const props = defineProps({
// })
const results = ref(undefined)

onMounted(async () => {
    getResults('recognition').then(r => results.value = r)
})
</script>

