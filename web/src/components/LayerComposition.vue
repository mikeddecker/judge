<template>
    <div v-if="composition">
        <h4 class="text-blue-700">{{ compositionName }} overview</h4>
        <div class="flex flex-wrap gap-4 mb-4">
            <LayerCompositionElementCard class="w-1/4" title="General" :element="composition['GeneralProperties']"></LayerCompositionElementCard>
            <LayerCompositionElementCard class="w-1/4" title="Start" :element="composition['StartProperties']"></LayerCompositionElementCard>
            <LayerCompositionElementCard class="w-1/4" title="End" :element="composition['EndProperties']"></LayerCompositionElementCard>
        </div>
        <h4>Stage properties</h4>
        <div class="flex flex-wrap gap-4 mb-4">
            <LayerCompositionElementCard 
                v-for="(stage, stageNr) in composition['StageProperties']" 
                :title="`stage ${stageNr}`" :element="stage" 
                class="w-1/5"></LayerCompositionElementCard>
        </div>
    </div>
        
    <h4>Add properties for {{ compositionName }}</h4>
    <div class="flex gap-2 mb-2">
        <Select v-model="selectedStage" :options="stages" option-label="label" option-value="value" placeholder="general"></Select>
        <Select v-model="selectedLayer" :options="layers" option-label="name" option-value="id" placeholder="select layer"></Select>
        <IftaLabel>
            <InputText :id="`customName-${compositionName}`" v-model="customName" variant="filled" :placeholder="selectedLayerName" />
            <label :for="`customName-${compositionName}`">Custom name</label>
        </IftaLabel>
        <Button v-if="selectedLayer" aria-label="Add composition" label="Add composition" icon="pi pi-database" @click="saveComposition"></Button>
    </div>
</template>

<script setup>
import { addLayerComposition, getLayers } from '@/services/videoService';
import { computed, onMounted, ref } from 'vue';
import LayerCompositionElementCard from './LayerCompositionElementCard.vue';

const props = defineProps({
    compositionName: {
        type: String,
        required: true
    },
    composition : {
        type: Object,
        required: false
    }
})

const emit = defineEmits(['composition-saved'])

const selectedStage = ref(null)
const selectedLayer = ref(null)
const customName = ref(null)
const layers = ref(null)

const stages = computed(() => {
    let s = [
        { label: 'general', value: null },
        { label: 'start', value: 0 },
        { label: 'end', value: -1 },
    ]
    let maxStage = props.composition ? Object.keys(props.composition['StageProperties']).length : 0
    for (let i = 1; i <= maxStage + 1; i++) {
        s.push({ label: `stage ${i}`, value: i })
    }
    console.log('max stage is', maxStage)
    return s
})

const selectedLayerName = computed(() => {
    if (!layers.value) { return 'Custom name' }
    return layers.value.find(layer => layer.id === selectedLayer.value)?.name || 'Select layer'
})

onMounted(async () => {
    getLayers().then(l => layers.value = l)
})

const saveComposition = async () => {
    if (!selectedLayer.value) { return }
    
    await addLayerComposition(props.compositionName, selectedStage.value, selectedLayer.value, customName.value)
    
    selectedLayer.value = null
    emit('composition-saved')
}
</script>

