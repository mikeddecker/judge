<script setup>
import { getVideoImagePath } from '@/services/videoService';
import { onMounted, ref } from 'vue';

let props = defineProps(['title', 'videoId'])

const imageUrl = ref('');

onMounted(async () => {
  try {
    imageUrl.value = await getVideoImagePath(props.videoId);
  } catch (error) {
    console.error('Error fetching image:', error);
  }
});
</script>

<template>
  <div class="videoinfo">
    <div class="container">
      <img v-if="imageUrl" :src="imageUrl" alt="Video thumbnail" />
      <p v-else>Loading image...</p>
    </div>
    <div class="info">
      <h2>{{ title }}</h2>
      <p>{{ videoId }}</p>
    </div>
  </div>
</template>

<style scoped>
.videoinfo {
  margin: 0.7%;
  padding: 0.2rem;
  width: 48%;
  border: 1px solid var(--color-border);
  border-radius: 0.55rem;
  box-shadow: 0.5px 0.5px 3px var(--color-heading);
}

.info {
  margin: 0 0.2rem;
}

h2 {
  margin-bottom: 0.4rem;
  color: var(--color-heading);
  word-wrap: break-word;
}

.container img {
  object-fit: contain;
  height: 100%;
  width: 100%;
}


@media (min-width: 1024px) {
  .videoinfo {
    width: 31%;
  }
}
</style>
