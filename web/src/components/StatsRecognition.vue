<template>
    <h2>Recognition stats</h2>
    <pre>{{ stats }}</pre>

</template>

<script>
import { onMounted } from 'vue'

const props = defineProps({
    stats: {
        type: Object,
        required: true,
    }
})
</script>

