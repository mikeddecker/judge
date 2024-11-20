<script setup>
defineProps(['title'])
</script>

<template>
  <div class="container">
    <div class="img">
      <img src="/home/miked/Pictures/Screenshots/burndown3.png">
    </div>
    <div class="info">
      <h2>{{ title }}</h2>
      <p>
        Some information
      </p>
    </div>
  </div>
</template>

<style scoped>
h2 {
  /* font-size: 1.2rem;
  font-weight: 500; */
  margin-bottom: 0.4rem;
  color: var(--color-heading);
}

@media (min-width: 1024px) {
}
</style>
