<script setup>
import { computed } from 'vue'

const props = defineProps({
  results: {
    type: Object,
    required: true,
  },
})

</script>


<template>
  <h2 class="mb-4">Judge scores</h2>
  <div class="flex gap-4">
    <pre>{{ results }}</pre>
  </div>
</template>

<style scoped>
.p-chart canvas {
  height: 30rem;
}

.p-card-body {
  padding-top: 0 !important;
}
</style>
