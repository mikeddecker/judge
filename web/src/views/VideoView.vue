<template>
  <div v-if="videoinfo">
    <h1>Label {{ videoinfo.Name }}</h1>
    <div v-if="loading">Loading...</div>
    <div v-if="error" class="error">{{ error }}</div>
    <VideoPlayer v-if="videoPath" v-bind:video-id="$route.params.id" :video-src="videoPath" :info="videoinfo"></VideoPlayer>
  </div>
  <div v-else>
    Loading ...
  </div>
</template>

<script>
import VideoPlayer from '@/components/VideoPlayer.vue';
import { getVideoInfo, getVideoPath } from '../services/videoService';

export default {
  components: {
    VideoPlayer,
  },
  data() {
    return {
      videoPath: null,
      loading: false,
      error: null,
      videoinfo: null,
    };
  },
  async created() {
    this.loading = true;
    try {
      this.videoPath = await getVideoPath(this.$route.params.id);
      this.videoinfo = await getVideoInfo(this.$route.params.id)
    } catch {
      this.error = 'Failed To load';
    } finally {
      this.loading = false;
    }
  },
};
</script>

<style scoped>
h1 {
  margin: 0.5rem 0
}
.error {
  color: red;
}
</style>
