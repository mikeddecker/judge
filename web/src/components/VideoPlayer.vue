<template>
  <p>LabeledFrames: {{ labeledFramesCount }} | Current frame : {{ currentFrame }} | FramesLabeledPerSecond : {{ framesLabeledPerSecond }}</p>
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
    <div class="controls">
      <button v-show="paused" @click="play">&#9654;</button>
      <button v-show="playing" @click="pause">&#9208;</button>
      <button @click="toggleLabelMode">Current modus: {{ labelMode }}</button>
      <div class="review-controls" v-show="modeIsReview">
        <button class="big-arrow" @click="setToPreviousFrameIdxAndDraw">&larr;</button>
        <button class="big-arrow" @click="setToNextFrameIdxAndDraw">&rarr;</button>
        <button @click="deleteLabel"><img src="@/assets/delete.png" alt="buttonpng" class="icon"/></button>
      </div>
    </div>
  </div>
  <pre>{{ vidinfo }}</pre>
</template>

<script setup>
import { getVideoInfo, postVideoFrame, removeVideoFrame } from '@/services/videoService';
import { computed, onBeforeMount, ref } from 'vue';

const props = defineProps(['title', 'videoId', 'videoSrc'])
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
const relativeWidth = computed(() => Math.abs(currentX.value - startX.value) / currentWidth.value)
const relativeHeight = computed(() => Math.abs(currentY.value - startY.value) / currentHeight.value)
const videoduration = ref(1)
const vidinfo = ref(null)
const labeledFramesCount = computed(() => vidinfo.value ? vidinfo.value.LabeledFrameCount : 0)
const labelMode = ref("localization")
const modeIsLocalization = computed(() => { return labelMode.value == "localization" })
const modeIsReview = computed(() => { return labelMode.value == "review" })
const currentFrameIdx = ref(0)
const framesLabeledPerSecond = computed(() => { return vidinfo.value ? vidinfo.value.FramesLabeledPerSecond.toFixed(2) : 0 })

function updatePaused(event) {
  videoduration.value = event.target.duration
  videoElement.value = event.target;
  paused.value = event.target.paused;
  currentFrame.value = Math.floor(modeIsLocalization.value ? vidinfo.value.FPS * event.target.currentTime : vidinfo.value.Frames[currentFrameIdx.value].FrameNr)
  console.log(currentFrame.value, vidinfo.value.FPS, event.target.currentTime)
  currentWidth.value = event.target.clientWidth
  currentHeight.value = event.target.clientHeight
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
function clearAndReturnCtx() {
  const ctx = canvas.value.getContext("2d");
  ctx.clearRect(0, 0, canvas.value.width, canvas.value.height);
  ctx.beginPath();
  return ctx
}
function startDrawing(event) {
  if (!modeIsLocalization.value) { return }
  if (playing.value) { pause() }
  isDrawing.value = true;
  startX.value = event.offsetX;
  startY.value = event.offsetY;
}
function drawRectangle(event) {
  if (!isDrawing.value) return;
  const ctx = clearAndReturnCtx()

  currentX.value = event.offsetX;
  currentY.value = event.offsetY;
  ctx.strokeStyle = 'lime';
  ctx.rect(startX.value, startY.value, currentX.value - startX.value, currentY.value - startY.value);
  ctx.stroke();
}
function setToPreviousFrameIdxAndDraw() {
  currentFrameIdx.value = currentFrameIdx.value - 1 < 0 ? labeledFramesCount.value - 1 : currentFrameIdx.value - 1
  drawCurrentFrameIdx()
}
function setToNextFrameIdxAndDraw() {
  currentFrameIdx.value = currentFrameIdx.value + 1 < labeledFramesCount.value ? currentFrameIdx.value + 1 : 0
  drawCurrentFrameIdx()
}
function drawCurrentFrameIdx() {
  const label = vidinfo.value.Frames[currentFrameIdx.value]
  const frameNr = label.FrameNr
  const ctx = clearAndReturnCtx()
  setCurrentTime(frameNr / vidinfo.value.FPS)
  ctx.strokeStyle = 'lime';
  const xleft = (label.X - label.Width / 2) * currentWidth.value
  const yleft = (label.Y - label.Height / 2) * currentHeight.value
  const w = label.Width * currentWidth.value
  const h = label.Height * currentHeight.value
  ctx.rect(xleft, yleft, w, h);
  ctx.stroke();
}
async function deleteLabel() {
  await removeVideoFrame(props.videoId, currentFrame.value).then(videoinfo => vidinfo.value = videoinfo).catch(err => console.error(err))
  setToNextFrameIdxAndDraw()
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
  let frameinfo = {
    "frameNr" : currentFrame.value,
    "x" : centerX.value, 
    "y" : centerY.value, 
    "width" : relativeWidth.value, 
    "height" : relativeHeight.value,
    "jumperVisible" : true
  }
  const fnr = currentFrame.value
  console.log(currentFrame.value)
  postVideoFrame(props.videoId, fnr, frameinfo).then(response => vidinfo.value=response.data)

  let frameNrAlreadyLabeled = true
  let rndTime = 0
  let rndFrameNr = 0
  while (frameNrAlreadyLabeled) {
    rndTime = Math.random() * videoduration.value
    rndFrameNr = Math.floor(rndTime * vidinfo.value.FPS)
    frameNrAlreadyLabeled = vidinfo.value.Frames.map(frameinfo => frameinfo.FrameNr).includes(rndFrameNr)
  }
  setCurrentTime(rndTime)
}
function toggleLabelMode() {
  if (modeIsLocalization.value) {
    labelMode.value = "review"
    pause()
    drawCurrentFrameIdx()
  } else if (modeIsReview.value) {
    labelMode.value = "localization"
    clearAndReturnCtx()
  }
}
onBeforeMount(async () => {
    vidinfo.value = await getVideoInfo(props.videoId);
})
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

.controls {
  margin: 0.3rem
}

.review-controls {
  margin: 0.2rem 0;
}

button {
  min-width: 5rem;
  height: 3rem;
  margin: 0.1rem
}

img {
  max-width: 50%;
  max-height: 50%;
}

.overlay-canvas {
  position: absolute;
  border: 1px solid red;
  top: 0;
  left: 0;
  /* width: 100%; */
  /* height: 100%; */
}

.big-arrow {
  font-size: 1.75rem;
}

.material-icons {
  font-size: 1.36rem
}

@media (min-width: 1024px) {
  video {
    max-height: 85vh;
  }
}
</style>
