<template>
    <h3>Validation results</h3>
    <ConfusionMatrix
        v-for="(matrix, layername) in confusionMatrices"
        :name="layername" :values="matrix"
        :headers="confusionHeaders[layername]"
    >
    </ConfusionMatrix>
      <!-- <ConfusionMatrix
    v-for="(matrix, prop) in results['models'][selectedModel]['validation_results']['metrics']['confusion']"
    :name="prop" :values="matrix"
    :headers="results['models'][selectedModel]['validation_results']['confusion_heads'][prop]"
  ></ConfusionMatrix> -->

    <pre>{{ trainresult }}</pre>
</template>

<script setup>
import ConfusionMatrix from './ConfusionMatrix.vue';
import { computed } from 'vue';

const props = defineProps({
    trainresult : {
        type: Object,
        required: true
    }
})

const validationResults = computed(() => props.trainresult.epochs[props.trainresult.bestEpoch])
const confusionMatrices = computed(() => validationResults.value.validationResults.metric_per_prop.confusion)
const confusionHeaders = computed(() => validationResults.value.validationResults.confusion_heads)

</script>

