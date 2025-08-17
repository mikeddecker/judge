<template>
    <div class="flex-auto">
        <h5>{{ title }}</h5>
        <Listbox :options="Object.keys(stageProperties)" listStyle="max-height:750px" disabled/>
        <LayerCompositionPropertyValueSelector
            v-for="(composition, compositionKey) in stageProperties"
            :property="composition['property']"
            :name="compositionKey"
            :value="composition['defaultValue']"
            @update:value="value => updateValue(composition, compositionKey, value)"
        ></LayerCompositionPropertyValueSelector>
    </div>
</template>

<script setup>
import LayerCompositionPropertyValueSelector from './LayerCompositionPropertyValueSelector.vue';

const props = defineProps({
    stageProperties: {
        type: Object,
        required: true
    },
    title: {
        type: String,
        required: true
    }
})

const emit = defineEmits(['update:value'])

const updateValue = (composition, key, value) => {
    composition['defaultValue'] = value
    emit('update:value', key, 'defaultValue', value)
}
</script>

