<script>
import { getVideoInfo, getVideoPath } from '../services/videoService';
// import { VidStack } from 'vuepress-plugin-components'; // Import VidStack here

// import { VueStack } from 'vuepress-plugin-components' 
// import { VidStack } from '@vidstack/player'; // Import VidStack here

export default {
  data() {
    return {
      data: null,
      loading: false,
      error: null,
      videoinfo: null,
    };
  },
  async created() {
    this.loading = true;
    try {
      this.data = await getVideoPath(this.$route.params.id);
      this.videoinfo = await getVideoInfo(this.$route.params.id)
    } catch {
      this.error = 'Failed To load';
    } finally {
      this.loading = false;
    }
  },
};
</script>

<template>
  <div v-if="videoinfo">
    <h1>Label {{ videoinfo.Name }}</h1>
    <div v-if="loading">Loading...</div>
    <div v-if="error" class="error">{{ error }}</div>
    <div v-if="data">
      <!-- <VidStack
        src="https://files.vidstack.io/sprite-fight/720p.mp4"
        poster="https://files.vidstack.io/sprite-fight/poster.webp"
      /> -->
      <pre>{{ data }}</pre>
      <p>{{ $route.params.id }}</p>
    </div>
  </div>
  <div v-else>
    Loading ...
  </div>
</template>


<style scoped>
.error {
  color: red;
}
</style>
