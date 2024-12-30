<template>
  <div class="container">
    <video class="absolute"
    ref="videoPlayer" :src="videoSrc"
    controls autoplay loop 
    @canplay="updatePaused" @playing="updatePaused" @pause="updatePaused"
    >
    </video>
    <canvas 
      ref="canvas" 
      :width="currentWidth" 
      :height="currentHeight" 
      class="overlay-canvas"
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
const centerX = computed(() => (startX.value + currentX.value) / 2.0 / currentWidth.value)
const centerY = computed(() => (startY.value + currentY.value) / 2.0 / currentHeight.value)
const relativeWidth = computed(() => (currentX.value - startX.value) / currentWidth.value)
const relativeHeight = computed(() => (currentY.value - startY.value) / currentHeight.value)
const videoduration = ref(1)

defineProps(['title', 'videoId', 'videoSrc'])
function updatePaused(event) {
  console.log("updatePaused", event)
  videoduration.value = event.target.duration
  console.log("video duration", event.target.duration)
  videoElement.value = event.target;
  paused.value = event.target.paused;
  currentFrame.value = Math.floor(29.97 * event.target.currentTime)
  currentWidth.value = event.target.clientWidth
  currentHeight.value = event.target.clientHeight
  console.log("current time", currentFrame.value)
}
function play() {
  videoElement.value.play();
}
function pause() {
  videoElement.value.pause();
}
function setCurrentTime(val) {
  videoElement.value.currentTime = val
}
function startDrawing(event) {
  isDrawing.value = true;
  startX.value = event.offsetX;
  startY.value = event.offsetY;
}
function drawRectangle(event) {
  if (!isDrawing.value) return;
  
  const ctx = canvas.value.getContext("2d");
  currentX.value = event.offsetX;
  currentY.value = event.offsetY;
  
  ctx.strokeStyle = 'lime';
  ctx.clearRect(0, 0, canvas.value.width, canvas.value.height);
  ctx.beginPath();
  ctx.rect(startX.value, startY.value, currentX.value - startX.value, currentY.value - startY.value);
  ctx.stroke();
}
function endDrawing(event) {
  if (!isDrawing.value) return;
  isDrawing.value = false;
  const ctx = canvas.value.getContext("2d");
  currentX.value = event.offsetX;
  currentY.value = event.offsetY;
  
  ctx.strokeStyle = 'lime';
  ctx.clearRect(0, 0, canvas.value.width, canvas.value.height);
  ctx.beginPath();
  ctx.rect(startX.value, startY.value, currentX.value - startX.value, currentY.value - startY.value);
  ctx.stroke();
  console.log("event at end is", startX.value, startY.value, currentX.value, currentY.value);
  console.log("LocationLabel is", centerX.value, centerY.value, relativeWidth.value, relativeHeight.value);
  setCurrentTime(Math.random() * videoduration.value)
}
// TODO : catch resize of window, because the current frame can be labeled wrong, position wise
</script>

<style scoped>
.container {
  position: relative;
  display: flex;
  justify-content: left;
  flex-wrap: wrap;
  max-width: 100%;
}

video {
  max-width: 100%;
  max-height: 70vh;
}

.overlay-canvas {
  position: absolute;
  border: 1px solid red;
  top: 0;
  left: 0;
  /* width: 100%; */
  /* height: 100%; */
}

@media (min-width: 1024px) {
  video {
    max-height: 85vh;
  }
}
</style>
