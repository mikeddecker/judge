<template>
  <div>
    <!-- <video v-if="mode == 'LOCALIZE'"
      id="vid" ref="videoPlayer" :src="videoSrc" loop
      @play="updatePlaying" @pause="updatePaused" @seeked="onSeeked" @ontimeupdate="ontimeupdate" @loadeddata="onLoadedData"
    /> -->
    <video
      id="vid" ref="videoPlayer" :src="videoSrc" loop controls
      @play="updatePlaying" @pause="updatePaused" @seeked="onSeeked" @timeupdate="ontimeupdate" @loadeddata="onLoadedData"
    />
    <canvas
      v-show="mode == 'LOCALIZE'"
      ref="canvas" 
      :width="videoWidth" 
      :height="videoHeight" 
      :class="mouse"
      @mousedown="canvasMouseDown" 
      @mousemove="canvasMouseMoves" 
      @mouseup="canvasMouseEndDrawing"
      @mouseleave="canvasMouseEndDrawing"
      >
        Your browser does not support the HTML canvas tag.
    </canvas>
    {{ boxesHovering }}
  </div>
</template>

<script setup>
import { onMounted, ref, computed, watch } from 'vue'

const props = defineProps(['title', 'videoId', 'videoSrc', 'mode', 'canvasMode', 'currentFrameNr', 'videoinfo'])
const emit = defineEmits(['play', 'pause', 'seeked', 'timeupdate', 'loadeddata'])
const videoElement = ref(null)
const videoWidth = ref(0)
const videoHeight = ref(0)

const ctx = ref(null)
const canvas = ref(null)
const mouse = ref('')

const modeIsLocalization = computed(() => props.mode == 'LOCALIZE')

const canvasmodeIsDraw = computed(() => props.canvasMode == 'draw')
const canvasmodeIsEdit = computed(() => props.canvasMode == 'edit')
const canvasmodeIsDelete = computed(() => props.canvasMode == 'delete')
const boxes = computed(() => props.videoinfo.Frames)
const boxesHovering = ref([])

const boxColors = [
  '#bfdbfe',
  '#fef9c3',
  '#fdba74',
  '#ec4899',
  '#bbf7d0',
  '#fee2e2',
  '#f3f4f6',
  '#67e8f9',
  '#dc2626',
  '#a8a29e'
]

const mouseX = ref(0)
const mouseY = ref(0)
const mouseXstart = ref(0)
const mouseYstart = ref(0)


onMounted(async () => {
  videoElement.value = document.getElementById("vid")
  canvas.value = document.getElementById("canvas") 
  console.log(props.mode == 'LOCALIZE')
})

watch(() => props.mode, (newMode, oldMode) => {
  // On Localize add or remove controls of video
  newMode == 'LOCALIZE' ? videoElement.value.removeAttribute('controls') : videoElement.value.setAttribute('controls', 'controls')

  // On Localize, set currentFrame to first boxes if available
})
watch(() => props.currentFrameNr, (newFrameNr, oldFrameNr) => {
  console.log(videoElement.value.paused, newFrameNr == Math.absoldFrameNr, newFrameNr, oldFrameNr)
  if (modeIsLocalization.value) {
    videoElement.value.currentTime = newFrameNr / props.videoinfo.FPS
    resetCanvasAndDrawBoxes()
  }
})

onMounted(async () => {

})
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
  emit('timeupdate', event.target.currentTime)  
}
function onLoadedData(event) {
  console.log("loadeddata", event)
  videoWidth.value = videoElement.value.clientWidth
  videoHeight.value = videoElement.value.clientHeight
}
onresize = (event) => {
  videoWidth.value = videoElement.value.clientWidth
  videoHeight.value = videoElement.value.clientHeight
}

/* =============
** Box Functions
============= */
const resetCanvasAndDrawBoxes = () => {
  let ctx = canvas.value.getContext("2d")
  ctx.clearRect(0, 0, canvas.value.width, canvas.value.height);
  //   ctx.beginPath();

  let boxes = filterBoxes(props.currentFrameNr)
  Object.entries(boxes).forEach(([idx, box]) => {

    console.log(idx, box)
    ctx.strokeStyle = boxColors[Number(idx) + 1]
    const xleft = (box.X - box.Width / 2) * videoWidth.value
    const yleft = (box.Y - box.Height / 2) * videoHeight.value
    const w = box.Width * videoWidth.value
    const h = box.Height * videoHeight.value
    ctx.strokeRect(xleft, yleft, w, h);
  })
  
  // Draw current drawing box
  console.log(mouseXstart.value, mouseYstart.value, mouseX.value - mouseXstart.value, mouseY.value - mouseYstart.value)
  ctx.strokeStyle = boxColors[0]
  ctx.strokeRect(mouseXstart.value * videoWidth.value, mouseYstart.value * videoHeight.value, (mouseX.value - mouseXstart.value) * videoWidth.value, (mouseY.value - mouseYstart.value) * videoHeight.value);

  console.log("drawing boxes are", boxes)
}


const canvasMouseDown = (event) => {
  if (canvasmodeIsDraw.value) {
    console.log("start drawing")
  }

  mouseXstart.value = event.offsetX / videoWidth.value;
  mouseYstart.value = event.offsetY / videoHeight.value;
  
}
const canvasMouseMoves = (event) => {
  console.log("mouse move", event)
  mouseX.value = event.offsetX / videoWidth.value;
  mouseY.value = event.offsetY / videoHeight.value;
  if (!canvasmodeIsDraw.value) {
    boxesHovering.value = boxes.value
    .filter(box => box.FrameNr == Math.round(props.currentFrameNr))
    .filter(box => {
      let minXbox = box.X - box.Width / 2
      let maxXbox = box.X + box.Width / 2
      let minYbox = box.Y - box.Height / 2
      let maxYbox = box.Y + box.Height / 2
      return minXbox < mouseX.value && mouseX.value < maxXbox && minYbox < mouseY.value && mouseY.value < maxYbox
    })
  }
  mouse.value = boxesHovering.value.length ? 'cursor-pointer' : canvasmodeIsDraw.value ? 'cursor-crosshair' : 'cursor-auto'

  // mouse.value = 'cursor-wait'
  
  resetCanvasAndDrawBoxes()
}
const canvasMouseEndDrawing = (event) => {
  mouse.value = ''
}

const filterBoxes = (frameNr) => {
  return props.videoinfo.Frames.filter((box) => box.FrameNr == Math.round(frameNr))
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
