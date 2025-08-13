<template>
    <div class="flex-auto">
        <h5>{{ title }}</h5>
        <Listbox :options="Object.keys(element)" listStyle="max-height:750px" disabled/>
    </div>
</template>

<script setup>

const props = defineProps({
    element: {
        type: Object,
        required: true
    },
    title: {
        type: String,
        required: true
    }
})
</script>

