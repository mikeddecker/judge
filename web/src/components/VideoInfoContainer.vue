<script setup>
import VideoInfo from './VideoInfo.vue';

defineProps(['title'])
</script>

<template>
    <div class="container">
      <VideoInfo title="Bye bye"/>
      <VideoInfo title="HELLOKES"/>
      <VideoInfo title="DD3"/>
      <VideoInfo title="Bye bye"/>
      <VideoInfo title="HELLOKES"/>
      <VideoInfo title="DD3"/>
      <VideoInfo title="Bye bye"/>
      <VideoInfo title="HELLOKES"/>
      <VideoInfo title="DD3"/>
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
