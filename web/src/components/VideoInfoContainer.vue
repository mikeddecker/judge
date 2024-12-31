<script setup>
import VideoInfo from './VideoInfo.vue';

defineProps(['title', 'videos', 'totalLabeledFrames'])

</script>

<template>
  <div class="container">
    <p>{{ totalLabeledFrames }}</p>
    <VideoInfo v-for="video in videos" :key="video.Id" :video-id="video.Id" :title="video.Name" :info="video"/>
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
