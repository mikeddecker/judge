<template>
    <div class="flex-auto border rounded-md m-2">
        <h5 class="m-3">{{ title }}</h5>
        <LayerCompositionValueSelector
            v-for="(composition, compositionKey) in stageProperties"
            :layer="composition['layer']"
            :name="compositionKey"
            :value="composition['defaultValue']"
            :focussed="composition['focussed']"
            @update:value="value => updateAttribute(composition, compositionKey, 'defaultValue', value)"
            @update:focussed="value => updateAttribute(composition, compositionKey, 'focussed', value)"
        ></LayerCompositionValueSelector>
    </div>
</template>

<script setup>
import LayerCompositionValueSelector from './LayerCompositionValueSelector.vue';

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

const emit = defineEmits(['update:attribute'])

const updateAttribute = (composition, key, attribute, value) => {
    composition[attribute] = value
    emit('update:attribute', key, attribute, value)
}
</script>

