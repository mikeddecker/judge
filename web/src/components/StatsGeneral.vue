<template>
    <h2>General stats</h2>
    <pre>{{ stats }}</pre>
</template>

<script setup>
const props = defineProps({
    stats: {
        type: Object,
        required: true,
    }
})
</script>

