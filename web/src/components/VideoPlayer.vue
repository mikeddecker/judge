<template>
  <div class="container">
    <video
      ref="videoPlayer" :src="videoSrc"
      controls autoplay loop 
      @canplay="updatePaused" @playing="updatePaused" @pause="updatePaused"
    />
    <div class="controls">
      <button v-show="paused" @click="play">&#9654;</button>
      <button v-show="playing" @click="pause">&#9208;</button>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const videoElement = ref(null)
const paused = ref(null)
const playing = computed(() => { return !paused.value})

defineProps(['title', 'videoId', 'videoSrc'])
function updatePaused(event) {
  videoElement.value = event.target;
  paused.value = event.target.paused;
}
function play() {
  videoElement.value.play();
}
function pause() {
  videoElement.value.pause();
}
</script>

<style scoped>
.container {
  display: flex;
  justify-content: left;
  flex-wrap: wrap;
  max-width: 100%;
}

video {
  max-width: 100%;
  max-height: 70vh;
}

@media (min-width: 1024px) {
  video {
    max-height: 100vh;
  }
}
</style>
