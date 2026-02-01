<template>
    <h1>Results</h1>
    <Message v-if="!results || Object.keys(results) == 0" severity="warn" variant="outlined" icon="pi pi-microchip-ai">
        Train a model first to see results
    </Message>
    <!-- TODO : update to component which uses list, tooltip description & @clickevent-->
    <ButtonContainer>
        <Button v-if="trainedRecipeCodes"
            v-for="recipeCode in trainedRecipeCodes"
            :class="selectedRecipe == recipeCode ? 'p-button-highlight': ''" :aria-label="recipeCode" :label="recipeCode"
            severity="success" variant="text" raised
            v-tooltip="`Display results for recipe ${recipeCode}`"
            @click="() => selectedRecipe = recipeCode"
        ></Button>
    </ButtonContainer>

    <TrainResultRecognition v-if="validationResults" :trainresult="validationResults"></TrainResultRecognition>
    <!-- + Model comparison -->
</template>

<script setup>

import ButtonContainer from '@/components/ButtonContainer.vue';
import TrainResultRecognition from '@/components/TrainResultRecognition.vue';
import { getResults } from '@/services/videoService';
import { computed, ref, onMounted } from 'vue'

// const props = defineProps({
// })
const results = ref(undefined)
const trainedRecipeCodes = ref(undefined)
const selectedRecipe = ref(undefined)
const validationResults = computed(() => {
    if (!selectedRecipe.value) { return undefined }
    return results.value[selectedRecipe.value]
})

onMounted(async () => {
    getResults('recognition').then(
        r => results.value = Object.fromEntries(
            Object.entries(r).map(
                ([i, recipeResults]) => [recipeResults.recipeCode, recipeResults]
            )
        )
    ).then(
        () => {
            trainedRecipeCodes.value =  Object.entries(results.value).map(([listindex, trainresult]) => {
                if (trainresult.isBestOfAll) { selectedRecipe.value = trainresult.recipeCode }
                return trainresult.recipeCode
            })
        }
    )
})
</script>

