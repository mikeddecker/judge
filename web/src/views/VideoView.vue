<template>
  <div v-if="videoinfo">
    <h1>Label {{ videoinfo.Name }}</h1>
    <div v-if="loading">Loading...</div>
    <div v-if="error" class="error">{{ error }}</div>
    <VideoPlayer v-if="!loading" v-bind:video-id="route.params.id" :video-src="videoPath"></VideoPlayer>
  </div>
  <div v-else>
    Loading...
  </div>
</template>

<script setup>
import VideoPlayer from '@/components/VideoPlayer.vue';
import { getVideoInfo, getVideoPath } from '../services/videoService';
import { onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router';

const route = useRoute()

const loading = ref(false)
const error = ref('')
const videoinfo = ref({})
const videoId = ref(route.params.id)
const videoPath = ref('')

watch(
  () => route.params.id,
  (newId) => (
    loadVideo(newId)
  )
)

onMounted(async () => {
  await loadVideo(videoId.value)
})
async function loadVideo(id) {
  loading.value = true;
  try {
    videoPath.value = await getVideoPath(id);
    videoinfo.value = await getVideoInfo(id)
  } catch {
    error.value = 'Failed To load';
  } finally {
    loading.value = false;
  }
}
</script>

<style scoped>
h1 {
  margin: 0.5rem 0 0 0
}
.error {
  color: red;
}
</style>
