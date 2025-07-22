<template>
    <h1>Train page</h1>
    
    <h2>Localize</h2>
    <div class="flex flex-wrap gap-6 my-4">
        <CardRecipe v-for="(recipe, modelname) in recipesLocalize" :title="modelname" step="LOCALIZE" :recipe="recipe"></CardRecipe>
    </div>
    
    <h2>Segment</h2>
    <div class="flex flex-wrap gap-6 my-4">
        <CardRecipe v-for="(recipe, modelname) in recipesSegment" :title="modelname" step="SEGMENT" :recipe="recipe"></CardRecipe>
    </div>
    
    <h2>Recognize</h2>
    <div class="flex flex-wrap gap-6 my-4">
        <CardRecipe v-for="(recipe, modelname) in recipesRecognize" :title="modelname" step="RECOGNIZE" :recipe="recipe"></CardRecipe>
    </div>
</template>

<script setup>
import CardRecipe from '@/components/CardRecipe.vue';
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
