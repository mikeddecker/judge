<template>
  <div>
    <h1>API Data</h1>
    <div v-if="loading">Loading...</div>
    <div v-if="error" class="error">{{ error }}</div>

    <div>
      <p>Select a videoId & frameNr to create an image from</p>
      <input v-model="videoId">
      <input v-model="frameNr">
      <button @click="createImage">create</button>
      <p>{{ antwoord }}</p>
    </div>
    <div v-if="data">
      <pre>{{ data }}</pre>
    </div>
  </div>
</template>

<script setup>
import { getFolder } from '../services/videoService';
import { onMounted, ref } from 'vue';
import { createVideoImage } from '@/services/videoService';


const data = ref(null)
const loading = ref(true)
const error = ref('')
const videoId = ref(0)
const frameNr = ref(0)
const antwoord = ref('')

onMounted(async () => {
  loading.value = true;
  try {
    data.value = await getFolder(1);
  } catch {
    error.value = 'Failed To load';
  } finally {
    loading.value = false;
  }
})

async function createImage() {
  createVideoImage(videoId.value, frameNr.value).then(response => {
    antwoord.value = response.data
  })
}

</script>

<style scoped>
.error {
  color: red;
}
</style>
