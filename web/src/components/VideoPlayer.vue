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
  </div>
</template>

<script setup>
import { onMounted, ref, computed, watch } from 'vue'

// Watch out, frame numbers can get floated: e.g. 112.000000000000001
const props = defineProps(['title', 'videoId', 'videoSrc', 'mode', 'canvasMode', 'currentFrameNr', 'videoinfo', 'labeltype'])
const emit = defineEmits(['play', 'pause', 'seeked', 'timeupdate', 'loadeddata', 'deleteBox', 'addBox'])
const videoElement = ref(null)
const videoWidth = ref(0)
const videoHeight = ref(0)

const ctx = ref(null)
const canvas = ref(null)
const mouse = ref('')

const modeIsLocalization = computed(() => props.mode == 'LOCALIZE')
const modeIsSkills = computed(() => props.mode == 'SKILLS')

const canvasmodeIsDraw = computed(() => props.canvasMode == 'draw')
const canvasmodeIsEdit = computed(() => props.canvasMode == 'edit')
const canvasmodeIsDelete = computed(() => props.canvasMode == 'delete')
const canvasmodeIsPredict = computed(() => props.canvasMode == 'predict')
const boxes = computed(() => props.videoinfo.Frames.filter(b => b.LabelType == props.labeltype))
const boxesHovering = ref([])
const selectedBox = ref(null)
const paused = computed(() => videoElement.value?.paused)

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
const isDrawing = ref(false)


onMounted(async () => {
  videoElement.value = document.getElementById("vid")
  canvas.value = document.getElementById("canvas") 
})

watch(() => props.mode, (newMode, oldMode) => {
  // On Localize add or remove controls of video
  newMode == 'LOCALIZE' ? videoElement.value.removeAttribute('controls') : videoElement.value.setAttribute('controls', 'controls')

  // On Localize, set currentFrame to first boxes if available
})
watch(() => props.currentFrameNr, (newFrameNr, oldFrameNr) => {
  if (modeIsLocalization.value) {
    videoElement.value.currentTime = newFrameNr / props.videoinfo.FPS
    resetCanvasAndDrawBoxes()
  }
  if (modeIsSkills.value && videoElement.value.paused) {
    videoElement.value.currentTime = newFrameNr / props.videoinfo.FPS
  }
})

onMounted(async () => {

})
/* =========
** Functions
========= */

function updatePlaying(event) {
  // console.log("updatePlaying", event)
}
function updatePaused(event) {
  emit('pause', event.target.currentTime)
}
function onSeeked(event) {
  // videoElement.value.pause()
}
function ontimeupdate(event) {
  emit('timeupdate', event.target.currentTime)  
}
function onLoadedData(event) {
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

    ctx.strokeStyle = boxColors[Number(idx) + 1]
    const xleft = (box.X - box.Width / 2) * videoWidth.value
    const yleft = (box.Y - box.Height / 2) * videoHeight.value
    const w = box.Width * videoWidth.value
    const h = box.Height * videoHeight.value
    ctx.strokeRect(xleft, yleft, w, h);
  })
  
  // Draw current drawing box
  if (!canvasmodeIsDelete.value && !canvasmodeIsPredict.value) {
    ctx.strokeStyle = boxColors[0]
    ctx.strokeRect(mouseXstart.value * videoWidth.value, mouseYstart.value * videoHeight.value, (mouseX.value - mouseXstart.value) * videoWidth.value, (mouseY.value - mouseYstart.value) * videoHeight.value);
  }
}


const canvasMouseDown = (event) => {
  if (canvasmodeIsDraw.value) { 
    isDrawing.value = true 
    mouseXstart.value = event.offsetX / videoWidth.value;
    mouseYstart.value = event.offsetY / videoHeight.value;
  }
}

const canvasMouseMoves = (event) => {
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
  mouseX.value = event.offsetX / videoWidth.value;
  mouseY.value = event.offsetY / videoHeight.value;

  if (canvasmodeIsDelete.value) {
    boxesHovering.value.forEach(box => emit("deleteBox", box))
  }

  if (canvasmodeIsDraw.value && isDrawing.value) {
    isDrawing.value = false
    let box = {
      "frameNr" : Math.round(props.currentFrameNr),
      "x" : (mouseXstart.value + mouseX.value) / 2,
      "y" : (mouseYstart.value + mouseY.value) / 2,
      "width" : Math.abs(mouseXstart.value - mouseX.value),
      "height" : Math.abs(mouseYstart.value - mouseY.value),
      "jumperVisible" : true,
      "labeltype" : 2
    }

    if (box['height'] > 0.03) {
      emit('addBox', box)
    }

    mouseXstart.value = 0
    mouseYstart.value = 0
  }
}

const filterBoxes = (frameNr) => {
  return boxes.value.filter((box) => box.FrameNr == Math.round(frameNr))
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
