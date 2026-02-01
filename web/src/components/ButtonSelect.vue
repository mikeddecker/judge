<template>
    <ButtonContainer v-if="items">
        <Button
            v-for="item in items"
            :class="selected == item ? 'p-button-highlight': ''" :aria-label="item" :label="item"
            severity="success" variant="text" raised
            :v-tooltip="tooltip"
            @click="$emit('update:selectedItem', item)"
        ></Button>
    </ButtonContainer>
</template>

<script setup>
import ButtonContainer from './ButtonContainer.vue';

const emit = defineEmits(['update:selectedItem'])
const props = defineProps({
    items : {
        type: Array,
        required: true
    },
    selected: {
        required: true
    },
    buttonSize: {
        type: String,
        required: false,
        default: 'small'
    },
    tooltip: {
        type: String,
        required: false,
        default: 'Select this list item'
    }
})
</script>