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
        <Select v-model="selectedStage" :options="numericStages" option-label="label" option-value="value" placeholder="general"></Select>
        <Select v-model="selectedLayer" :options="layers" option-label="name" option-value="id" placeholder="select layer"></Select>
        <IftaLabel>
            <InputText :id="`customName-${compositionName}`" v-model="customName" variant="filled" :placeholder="selectedLayerName" />
            <label :for="`customName-${compositionName}`">Custom name</label>
        </IftaLabel>
        <Button v-if="selectedLayer" aria-label="Add composition" label="Add composition" icon="pi pi-database" @click="saveComposition"></Button>
    </div>
        
    <h4 v-if="composition">Move properties for {{ compositionName }}</h4>
    <div v-if="composition" class="flex gap-2 mb-2">
        <Select v-model="selectedSourceStage" :options="stages" placeholder="source"></Select>
        <Select v-model="selectedDestStage" :options="stages" placeholder="destination"></Select>
        <Select v-if="selectedSourceOrDestIsStageProperties" v-model="selectedStageNr" :options="possibleStageNrsToMoveToOrFrom" placeholder="select stageNr"></Select>
        <Select v-if="sourceFilterVisible" v-model="selectedLayerToMove" :options="sourceFilterdLayers" placeholder="select layer"></Select>
        <Button 
            v-if="selectedSourceStage && selectedDestStage && selectedLayerToMove && ((selectedSourceOrDestIsStageProperties && selectedStageNr) || !selectedSourceOrDestIsStageProperties)" 
            aria-label="Move property" label="Move property" icon="pi pi-move" 
            @click="moveProperty"
        ></Button>
    </div>
</template>

<script setup>
import { addLayerComposition, getLayers, moveLayerProperty } from '@/services/videoService';
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

const emit = defineEmits(['composition-saved', 'moved:property'])

// Add
const selectedStage = ref(null)
const selectedLayer = ref(null)

// Move
const selectedSourceStage = ref(null)
const selectedDestStage = ref(null)
const selectedLayerToMove = ref(null)
const selectedStageNr = ref(null)

const customName = ref(null)
const layers = ref(null)
const sourceFilterVisible = computed(() => (selectedSourceStage.value == 'StageProperties' && selectedStageNr.value) || (selectedSourceStage.value && selectedSourceStage.value != 'StageProperties'))
const sourceFilterdLayers = computed(() => {
    console.log(selectedSourceStage.value == 'StageProperties' ? Object.keys(props.composition[selectedSourceStage.value][selectedStageNr]) : Object.keys(props.composition[selectedSourceStage.value]))
    return selectedSourceStage.value == 'StageProperties' ? Object.keys(props.composition[selectedSourceStage.value][selectedStageNr]) : Object.keys(props.composition[selectedSourceStage.value])
})
const selectedSourceOrDestIsStageProperties = computed(() => selectedDestStage.value == 'StageProperties' || selectedSourceStage.value == 'StageProperties')
const possibleStageNrsToMoveToOrFrom = computed(() => {

})
const stages = ['GeneralProperties', 'StartProperties', 'EndProperties', 'StageProperties']
const numericStages = computed(() => {
    let s =  [
        { label: 'GeneralProperties', value: null },
        { label: 'StartProperties', value: 0 },
        { label: 'EndProperties', value: -1 },
    ]
    let maxStage = props.composition ? Object.keys(props.composition['StageProperties']).length : 0
    for (let i = 1; i <= maxStage + 1; i++) {
        s.push({ label: `stage ${i}`, value: i })
    }
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

const moveProperty = async () => {
    await moveLayerProperty(
        props.compositionName, 
        selectedLayerToMove.value, 
        selectedSourceStage.value, 
        selectedDestStage.value, 
        selectedSourceOrDestIsStageProperties.value ? selectedStageNr.value : null
    )
    selectedLayerToMove.value = null
    selectedSourceStage.value = null
    selectedDestStage.value = null
    selectedStageNr.value = null
    emit('moved:property')
}
</script>

