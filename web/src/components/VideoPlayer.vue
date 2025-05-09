<template>
  <div>
    <!-- <video v-if="mode == 'LOCALIZE'"
      id="vid" ref="videoPlayer" :src="videoSrc" loop
      @play="updatePlaying" @pause="updatePaused" @seeked="onSeeked" @ontimeupdate="ontimeupdate" @loadeddata="onLoadedData"
    /> -->
    <video
      id="vid" ref="videoPlayer" :src="videoSrc" loop controls :class="mode == 'LOCALIZE' ? 'pointer-events-none' : 'pointer-events-auto'"
      @play="updatePlaying" @pause="updatePaused" @seeked="onSeeked" @timeupdate="ontimeupdate" @loadeddata="onLoadedData"
    />
    <canvas
      v-show="mode == 'LOCALIZE'"
      ref="canvas" 
      :width="videoWidth" 
      :height="videoHeight" 
      @mousedown="startDrawing" 
      @mousemove="drawRectangle" 
      @mouseup="endDrawing"
      @mouseleave="endDrawing"
      >
        Your browser does not support the HTML canvas tag.
    </canvas>
  </div>
</template>

<script setup>
import { onMounted, ref, watch } from 'vue'

const props = defineProps(['title', 'videoId', 'videoSrc', 'mode', 'currentFrameNr', 'videoinfo'])
const emit = defineEmits(['play', 'pause', 'seeked', 'timeupdate', 'loadeddata'])
const videoElement = ref(null)
const videoWidth = ref(0)
const videoHeight = ref(0)

onMounted(async () => {
  videoElement.value = document.getElementById("vid")
  console.log(props.mode == 'LOCALIZE')
})

watch(() => props.mode, (newMode, oldMode) => {
  // On Localize add or remove controls of video
  newMode == 'LOCALIZE' ? videoElement.value.removeAttribute('controls') : videoElement.value.setAttribute('controls', 'controls')

  // On Localize, set currentFrame to first boxes if available
})
watch(() => props.currentFrameNr, (newFrameNr, oldFrameNr) => console.log(newFrameNr == Math.absoldFrameNr, newFrameNr, oldFrameNr))

/* =========
** Functions
========= */

function updatePlaying(event) {
  console.log("updatePlaying", event)
}
function updatePaused(event) {
  emit('pause', event.target.currentTime)
}
function onSeeked(event) {
  console.log("onSeeked", event)
}
function ontimeupdate(event) {
  console.log("upontimeupdate", event)
  emit('timeupdate', event.target.currentTime)
  
}
function onLoadedData(event) {
  console.log("loadeddata", event)
  console.log(videoElement)
  videoWidth.value = videoElement.value.clientWidth
  videoHeight.value = videoElement.value.clientHeight
}
onresize = (event) => {
  videoWidth.value = videoElement.value.clientWidth
  videoHeight.value = videoElement.value.clientHeight
}

</script>

<style scoped>
canvas {
  position: absolute;
  left: 0;
  top: 0;
}

@media (min-width: 1024px) {
}
</style>
