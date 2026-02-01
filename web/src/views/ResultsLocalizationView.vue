<template>
  <h1>Results</h1>
  <h2>Localization Recipes Performance</h2>
  <DataTable v-if="results" :value="Object.values(results)">
      <Column key="model" field="model" header="model" sortable></Column>
      <Column key="team_raw_avg" field="team_raw_avg" header="team_raw_avg" sortable><template #body="slotProps">{{ formatPercentage(slotProps.data['team_raw_avg']) }}</template></Column>
      <Column key="team_smoothing_avg" field="team_smoothing_avg" header="team_smoothing_avg" sortable><template #body="slotProps">{{ formatPercentage(slotProps.data['team_smoothing_avg']) }}</template></Column>
      <Column key="fitness" field="fitness" header="fitness" sortable><template #body="slotProps">{{ formatPercentage(slotProps.data['fitness']) }}</template></Column>
      <Column key="metrics/mAP50(B)" field="metrics/mAP50(B)" header="metrics/mAP50(B)" sortable><template #body="slotProps">{{ formatPercentage(slotProps.data['metrics/mAP50(B)']) }}</template></Column>
      <Column key="metrics/mAP50-95(B)" field="metrics/mAP50-95(B)" header="metrics/mAP50-95(B)" sortable><template #body="slotProps">{{ formatPercentage(slotProps.data['metrics/mAP50-95(B)']) }}</template></Column>
      <Column key="metrics/precision(B)" field="metrics/precision(B)" header="metrics/precision(B)" sortable><template #body="slotProps">{{ formatPercentage(slotProps.data['metrics/precision(B)']) }}</template></Column>
      <Column key="metrics/recall(B)" field="metrics/recall(B)" header="metrics/recall(B)" sortable><template #body="slotProps">{{ formatPercentage(slotProps.data['metrics/recall(B)']) }}</template></Column>
  </DataTable>

  <ButtonContainer>
    <Button v-if="recipes"
      :class="selectedRecipe == recipe ? 'p-button-highlight': ''" :aria-label="recipe" :label="recipe"
      severity="success" variant="text" raised
      v-tooltip="`Display results for recipe ${recipe}`"
      @click="() => selectedRecipe = recipe"
      v-for="recipe in Object.keys(recipes)"
    ></Button>
  </ButtonContainer>
  <ResultsLocalization v-if="selectedValidationResults" :results="selectedValidationResults"></ResultsLocalization>
  <h3>Recipes localize</h3>
  <pre>{{ recipes }}</pre>
</template>

<script setup>
import ButtonContainer from '@/components/ButtonContainer.vue';
import ResultsLocalization from '@/components/ResultsLocalization.vue';
import { formatPercentage } from '@/helpers/utils';
import { getJobOptions, getResults } from '@/services/videoService';
import { computed, onMounted, ref } from 'vue';

const recipes = ref(undefined)
const results = ref(undefined)
const selectedRecipe = ref(undefined)
const selectedValidationResults = computed(() => results.value ? results.value[selectedRecipe.value] : undefined)

onMounted(async () => {
  getJobOptions('LOCALIZE').then(r => recipes.value = r)
  getResults('localization').then(
    r => results.value = r
  ).then(
    // TODO : getLocalizeBoxMethods (smoothing types)
    () => selectedRecipe.value = Object.keys(results.value)[0]
  )
})
</script>

