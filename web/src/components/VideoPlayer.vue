<template>
  <div class="container">
    <video class="absolute"
      ref="videoPlayer" :src="videoSrc"
      controls autoplay loop 
      @canplay="updatePaused" @playing="updatePaused" @pause="updatePaused" @click="drawBox"
    ></video>
    <canvas 
      ref="canvas" 
      :width="currentWidth" :height="currentHeight" 
      style="border:1px solid #000000;"
      @mousedown="startDrawing"
      @mousemove="drawRectangle"
      @mouseup="endDrawing"
      @mouseleave="endDrawing"
    >
      Your browser does not support the HTML canvas tag.
    </canvas>
    <p>{{ currentFrame }}</p>
    <div class="controls">
      <button v-show="paused" @click="play">&#9654;</button>
      <button v-show="playing" @click="pause">&#9208;</button>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const videoElement = ref(null)
const canvas = ref(null)
const paused = ref(null)
const playing = computed(() => { return !paused.value})
const currentFrame = ref(0)
const currentWidth = ref(null)
const currentHeight = ref(null)
const isDrawing = ref(false)
const startX = ref(0)
const startY = ref(0)
const currentX = ref(0)
const currentY = ref(0)

defineProps(['title', 'videoId', 'videoSrc'])
function updatePaused(event) {
  // console.log("updatePaused", event)
  videoElement.value = event.target;
  paused.value = event.target.paused;
  currentFrame.value = Math.floor(29.97 * event.target.currentTime)
  currentWidth.value = event.target.clientWidth
  currentHeight.value = event.target.clientHeight
  // console.log("current time", currentFrame.value)
}
function play() {
  videoElement.value.play();
}
function pause() {
  videoElement.value.pause();
}
function drawBox(event) {
  // console.log("event", event)
  console.log("x, y", event.offsetX, event.offsetY, currentWidth.value, currentHeight.value)
}
function startDrawing(event) {
  isDrawing.value = true;
  console.log("event is", event)
  startX.value = event.offsetX;
  startY.value = event.offsetY;
}
function drawRectangle(event) {
  if (!isDrawing.value) return;
  
  const ctx = canvas.value.getContext("2d");
  currentX.value = event.offsetX;
  currentY.value = event.offsetY;
  
  ctx.clearRect(0, 0, canvas.value.width, canvas.value.height);
  ctx.beginPath();
  ctx.rect(startX.value, startY.value, currentX.value - startX.value, currentY.value - startY.value);
  ctx.stroke();
}
function endDrawing(event) {
  if (!isDrawing.value) return;
  isDrawing.value = false;
  console.log("event at end is", event)
  const ctx = canvas.value.getContext("2d");
  currentX.value = event.offsetX;
  currentY.value = event.offsetY;
  
  ctx.clearRect(0, 0, canvas.value.width, canvas.value.height);
  ctx.beginPath();
  ctx.rect(startX.value, startY.value, currentX.value - startX.value, currentY.value - startY.value);
  ctx.stroke();
}
// TODO : catch resize of window, because the current frame can be labeled wrong, position wise
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
    max-height: 85vh;
  }
}
</style>
