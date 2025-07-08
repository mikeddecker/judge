<script setup>
import { computed } from 'vue'

const props = defineProps({
  stats: {
    type: Object,
    required: true,
  },
})

</script>


<template>
  <pre>{{ stats }}</pre>
</template>

<style scoped>

</style>
