<template>
  <p>LabeledFrames: {{ labeledFramesCount }} | Current frame : {{ currentFrame }} | FramesLabeledPerSecond : {{ framesLabeledPerSecond }} | total labels: {{ totalLabels }} | Full box for all jumpers : {{ modeLocalizationIsAll }}</p>
  <button v-show="modeLocalizationIsAll" @click="toggleLocalizationType">1 box 4 all</button>
  <button v-show="!modeLocalizationIsAll" @click="toggleLocalizationType">1 box / jumper</button>
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
      <button v-show="paused && modeIsLocalization" @click="play">&#9654;</button>
      <button v-show="playing" @click="pause">&#9208;</button>
      <button @click="toggleLabelMode">Current modus: {{ labelMode }}</button>
      <button v-show="modeIsLocalization && modeLocalizationIsAll" @click="postFullFrameLabelAndDisplayNextFrame">label as full screen</button>
      <button v-show="modeIsLocalization" @click="displayNextRandomFrame">random next frame</button>
      <div class="review-controls" v-show="modeIsReview">
        <button class="big-arrow" @click="setToPreviousFrameAndDraw">&larr;</button>
        <button class="big-arrow" @click="setToNextFrameAndDraw">&rarr;</button>
        <button v-show="modeIsReview && modeLocalizationIsAll" @click="deleteLabel"><img src="@/assets/delete.png" alt="buttonpng" class="icon"/></button>
      </div>
      <BoxCard v-for="(box, index) in currentBoxes" :key="index" :frameinfo="box" @deleteBox="deleteLabel"/>
    </div>
  </div>
  <pre>{{ vidinfo }}</pre>
</template>

<script setup>
import { getVideoInfo, postVideoFrame, removeVideoFrame } from '@/services/videoService';
import { computed, onBeforeMount, onMounted, ref } from 'vue';
import { useRouter } from 'vue-router'
import BoxCard from './BoxCard.vue';

const props = defineProps(['title', 'videoId', 'videoSrc'])
const router = useRouter()
const colors = [
  "blue",
  "white",
  "pink",
  "yellow",
] // Each color here must have a class in box card!

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
const modeLocalizationIsAll = ref(false)
const currentFrameIdx = ref(0)
const framesLabeledPerSecond = computed(() => { return vidinfo.value ? vidinfo.value.FramesLabeledPerSecond.toFixed(2) : 0 })
const totalLabels = ref(0)
const avgLabels = ref(12)
const currentBoxes = ref([])

// Only for dd3 labeling
const videos = ref(null)
const nextVideoId = ref(props.videoId)
onMounted(async () => {
  // TODO : fix random labeling
  // console.log("mountin?")
  // getFolder(3).then((value) => {
  //   videos.value = Object.keys(value.Videos)
  //   console.log(videos.value)
  //   while (nextVideoId.value == props.videoId) {
  //     console.log("qsidm")
  //     let potentialNextVideoId = Number(videos.value[Math.floor(Math.random()*videos.value.length)])
  //     totalLabels.value = Object.values(value.Videos).reduce((prevValue, currentVideoInfo) => prevValue + currentVideoInfo.LabeledFrameCount, 0)
  //     avgLabels.value = totalLabels.value / Object.values(value.Videos).length
  //     let labeledFramesVideo = value.Videos[potentialNextVideoId].LabeledFrameCount
  //     if (labeledFramesVideo < avgLabels.value * 0.7 && potentialNextVideoId != 1208) {
  //       nextVideoId.value = potentialNextVideoId
  //     }
  //   }
  // })
})

function updatePaused(event) {
  videoduration.value = event.target.duration
  videoElement.value = event.target;
  if (!paused.value == true) {
    currentFrame.value = Math.floor(modeIsLocalization.value ? vidinfo.value.FPS * event.target.currentTime : vidinfo.value.Frames[currentFrameIdx.value].FrameNr)
  }
  paused.value = event.target.paused;
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
  if (modeLocalizationIsAll.value && !modeIsLocalization.value) { return }
  if (playing.value) { pause() }
  isDrawing.value = true;
  startX.value = event.offsetX;
  startY.value = event.offsetY;
}
function drawRectangle(event) {
  if (!isDrawing.value) return;
  const ctx = clearAndReturnCtx()
  
  drawCurrentFrame()
  currentX.value = event.offsetX;
  currentY.value = event.offsetY;
  ctx.strokeStyle = 'lime';
  ctx.rect(startX.value, startY.value, currentX.value - startX.value, currentY.value - startY.value);
  ctx.stroke();
}
function drawJumperRectangles() {
}
function setToPreviousFrameAndDraw() {
  let maxFrameNr = vidinfo.value.Frames.reduce((previous, current) => Math.max(previous, current.FrameNr), -Infinity)
  let smallerFrameNr = vidinfo.value.Frames
    .filter((frameinfo) => frameinfo.FrameNr < currentFrame.value)
    .reduce((previous, current) => Math.max(previous, current.FrameNr), -Infinity)
  currentFrame.value = smallerFrameNr == -Infinity ? maxFrameNr : smallerFrameNr
  setCurrentTime(currentFrame.value / vidinfo.value.FPS)

  drawCurrentFrame()
}
function setToNextFrameAndDraw() {
  let minFrameNr = vidinfo.value.Frames.reduce((previous, current) => Math.min(previous, current.FrameNr), Infinity)
  let biggerFrameNr = vidinfo.value.Frames
    .filter((frameinfo) => frameinfo.FrameNr > currentFrame.value)
    .reduce((previous, current) => Math.min(previous, current.FrameNr), Infinity)
  currentFrame.value = biggerFrameNr == Infinity ? minFrameNr : biggerFrameNr
  setCurrentTime(currentFrame.value / vidinfo.value.FPS)

  drawCurrentFrame()
}
function drawCurrentFrame() {
  const labels = vidinfo.value.Frames.filter((f) => f.FrameNr == currentFrame.value)
  const ctx = clearAndReturnCtx()
  for (let objKey in labels) {
    let label = labels[objKey]
    ctx.strokeStyle = colors[parseInt(objKey)];
    label.color = colors[objKey]
    const xleft = (label.X - label.Width / 2) * currentWidth.value
    const yleft = (label.Y - label.Height / 2) * currentHeight.value
    const w = label.Width * currentWidth.value
    const h = label.Height * currentHeight.value
    ctx.strokeRect(xleft, yleft, w, h);
  }
  currentBoxes.value = labels
}
async function deleteLabel(frameinfo) {
  await removeVideoFrame(props.videoId, currentFrame.value, frameinfo).then(videoinfo => vidinfo.value = videoinfo).catch(err => console.error(err))
  currentBoxes.value = currentBoxes.value.filter((f) => f != frameinfo)

  if (modeLocalizationIsAll.value) {
    setToNextFrameAndDraw()
  } else {
    drawJumperRectangles()
  }
}

function endDrawing(event) {
  if (!isDrawing.value) return;
  isDrawing.value = false;
  const ctx = canvas.value.getContext("2d");
  currentX.value = event.offsetX;
  currentY.value = event.offsetY;
  
  ctx.strokeStyle = 'lime';
  ctx.clearRect(0, 0, canvas.value.width, canvas.value.height);
  drawCurrentFrame()
  ctx.beginPath();
  ctx.strokeRect(startX.value, startY.value, currentX.value - startX.value, currentY.value - startY.value);
  let frameinfo = {
    "frameNr" : currentFrame.value,
    "x" : centerX.value, 
    "y" : centerY.value, 
    "width" : relativeWidth.value, 
    "height" : relativeHeight.value,
    "jumperVisible" : true,
    "labeltype" : modeLocalizationIsAll.value ? 1 : 2,
  }
  const fnr = currentFrame.value
  if (frameinfo['height'] > 0.07) {
    postVideoFrame(props.videoId, fnr, frameinfo).then(response => vidinfo.value=response.data)
  }
  
  currentBoxes.value.push(frameinfo)
}
function postFullFrameLabelAndDisplayNextFrame() {
  if (!modeLocalizationIsAll.value) {
    return
  }
  let frameinfo = {
    "frameNr" : currentFrame.value,
    "x" : 0.5, 
    "y" : 0.5, 
    "width" : 1.0, 
    "height" : 1.0,
    "jumperVisible" : true,
    "labeltype": 1,
  }
  const fnr = currentFrame.value
  postVideoFrame(props.videoId, fnr, frameinfo).then(response => vidinfo.value=response.data)

  displayNextRandomFrame()
}
function displayNextRandomFrame() {
  clearAndReturnCtx()
  if (Math.random() < framesLabeledPerSecond.value - avgLabels.value / 100) {
    router.push(`/browse`)
  } else {
    let frameNrAlreadyLabeled = true
    let rndTime = 0
    let rndFrameNr = 0
    while (frameNrAlreadyLabeled) {
      rndTime = Math.random() * videoduration.value
      rndFrameNr = Math.floor(rndTime * vidinfo.value.FPS)
      frameNrAlreadyLabeled = vidinfo.value.Frames.map(frameinfo => frameinfo.FrameNr).includes(rndFrameNr)
    }
    setCurrentTime(rndTime)
    currentFrame.value = rndFrameNr
  }
}
function toggleLabelMode() {
  if (modeIsLocalization.value) {
    labelMode.value = "review"
    pause()
    currentFrame.value = vidinfo.value.Frames[0].FrameNr
    setCurrentTime(currentFrame.value / vidinfo.value.FPS)
    drawCurrentFrame()
  } else if (modeIsReview.value) {
    labelMode.value = "localization"
    currentBoxes.value = []
    clearAndReturnCtx()
  }
}
function toggleLocalizationType() {
  modeLocalizationIsAll.value = !modeLocalizationIsAll.value
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
    max-height: 68vh;
  }
}
</style>
