<template>
    <h2>Segmentation stats</h2>
    <pre>{{ stats }}</pre>

</template>

<script>
const props = defineProps({
    stats: {
        type: Object,
        required: true,
    }
})
</script>

