<script setup>
defineProps({
  results: {
    type: Object,
    required: false,
  },
})
</script>

<template>
  <h2>Segmentation results</h2>
  <pre>{{ results }}</pre>
</template>

<style scoped>
</style>
