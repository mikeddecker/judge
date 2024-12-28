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
    <h1>Navigate videos</h1>
    <VideoInfoContainer/>
  </div>
</template>

<style>
@media (min-width: 1024px) {
  .flexcontainer {
    display: flex
  }
}
</style>
