<template>
    <h1>Train page</h1>
    
    <h2>Localize</h2>
    <pre>{{ recipesLocalize }}</pre>
    
    <h2>Segment</h2>
    <pre>{{ recipesSegment }}</pre>
    
    <h2>Recognize</h2>
    <pre>{{ recipesRecognize }}</pre>
</template>

<script setup>
import { getJobOptions } from '@/services/videoService';
import { onMounted, ref } from 'vue';

const recipesLocalize = ref(null)
const recipesSegment = ref(null)
const recipesRecognize = ref(null)

onMounted(async () => {
    getJobOptions('LOCALIZE').then(recipes => recipesLocalize.value = recipes)
    getJobOptions('SEGMENT').then(recipes => recipesSegment.value = recipes)
    getJobOptions('RECOGNIZE').then(recipes => recipesRecognize.value = recipes)
})
</script>
