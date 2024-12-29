<script setup>
import VideoInfo from './VideoInfo.vue';

defineProps(['title', 'videos'])
</script>

<template>
  <div class="container">
    <VideoInfo v-for="video in videos" :key="video.Id" :title="video.Name"/>
  </div>
</template>

<style scoped>
.container {
  display: flex;
  justify-content: left;
  flex-wrap: wrap;
}

@media (min-width: 1024px) {
}
</style>
