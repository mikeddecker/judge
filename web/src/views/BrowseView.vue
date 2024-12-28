<script>
import VideoInfoContainer from '@/components/VideoInfoContainer.vue';
export default {
  components: {
    VideoInfoContainer,
  },
};
</script>

<template>
  <div class="browse">
    <h1>This is the video browse page</h1>
    <VideoInfoContainer title="Bye bye"/>
    <VideoInfoContainer title="HELLOKES"/>
  </div>
</template>

<style>
@media (min-width: 1024px) {
  .flexcontainer {
    display: flex
  }
}
</style>
