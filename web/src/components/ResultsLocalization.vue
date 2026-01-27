<template>
    <ButtonContainer>
        <Button
            v-for="smoothTechnique in smoothTechniques" :label="smoothTechnique" :aria-label="smoothTechnique"
            severity="success" variant="text" raised size="small"
            :class="selectedSmoothTechnique == smoothTechnique ? 'p-button-highlight' : ''"
            @click="smoothTechniqueChosenValue = smoothTechnique"
        ></Button>
    </ButtonContainer>

    <h3>Performance per video</h3>
    <TableIous :ious="tabledIous"></TableIous>
</template>

<script setup>
import { formatPercentage } from '@/helpers/utils';
import TableIous from './TableIous.vue';
import { computed, onMounted, ref } from 'vue';
import ButtonContainer from './ButtonContainer.vue';

const props = defineProps({
    results: {
        type: Object,
        required: true,
    }
})

const smoothTechniques = computed(() => Object.keys(props.results['ious']))

const smoothTechniqueDefaultValue = computed(() => Object.keys(props.results['ious'])[0])
const smoothTechniqueChosenValue = ref(null)
const selectedSmoothTechnique = computed(() => smoothTechniqueChosenValue.value || smoothTechniqueDefaultValue.value )

const tabledIous = computed(() => {
    return Object.fromEntries(
        Object.entries(props.results['ious'][selectedSmoothTechnique.value]['val']['videos']).map(
            ([videoId, scores]) => [videoId, {
                videoId: Number(videoId),
                ...scores,
                labels: scores.ious.length
            }]
        )
    )
})

</script>

