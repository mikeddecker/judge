<template>
  <h2>Localize</h2>
  <pre>{{ recipesLocalize }}</pre>
</template>

<script setup>
import { getJobOptions } from '@/services/videoService';
import { onMounted, ref } from 'vue';

const recipesLocalize = ref(null)

onMounted(async () => {
    getJobOptions('LOCALIZE').then(recipes => recipesLocalize.value = recipes)
})
</script>
