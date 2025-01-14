<template>
  <div>
    <h1>Download page</h1>
    <div v-if="loading">Loading...</div>
    <div v-if="error" class="error">{{ error }}</div>
    <p>Provide a youtube url or youtube id</p>
    <input v-model="message" :placeholder="message" />
    <YTplayer :video-src="message"></YTplayer>

    <p>{{ message }}</p>
  </div>
</template>

<script>
import YTplayer from '@/components/YTplayer.vue';

export default {
  components: {
    YTplayer,
  },
  data() {
    return {
      videoPath: null,
      loading: false,
      error: null,
      videoinfo: null,
      message: "https://www.youtube.com/watch?v=oRdxUFDoQe0" // Beat it Michael Jackson
    };
  },
  async created() {
    this.loading = true;
    try {
      this.videoPath = this.message;
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
input {
  width: 500px;
}
</style>
