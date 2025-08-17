<template>
  <div>
    <video
      id="vid" ref="videoPlayer" :src="videoSrc" loop controls class="max-h-[98vh]"
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
      @mouseleave="canvasMouseLeave"
      >
        Your browser does not support the HTML canvas tag.
    </canvas>
  </div>
</template>

<script setup>
import { onMounted, ref, computed, watch } from 'vue'

// Watch out, frame numbers can get floated: e.g. 112.000000000000001
const props = defineProps(['title', 'videoId', 'videoSrc', 'mode', 'canvasMode', 'currentFrameNr', 'videoinfo', 'labeltype', 'predictedBoxes', 'drawPredictedBoxes'])
const emit = defineEmits(['play', 'pause', 'seeked', 'timeupdate', 'loadeddata', 'deleteBox', 'addBox'])
const videoElement = ref(null)
const videoWidth = ref(0)
const videoHeight = ref(0)

const canvas = ref(null)
const mouse = ref('')

const modeIsLocalization = computed(() => props.mode == 'LOCALIZE')
const modeIsSkills = computed(() => props.mode == 'SKILLS')

const canvasmodeIsDraw = computed(() => props.canvasMode == 'draw')
const canvasmodeIsEdit = computed(() => props.canvasMode == 'edit')
const canvasmodeIsDelete = computed(() => props.canvasMode == 'delete')
const canvasmodeIsAcceptPredictedBox = computed(() => props.canvasMode == 'accept')
const boxes = computed(() => props.videoinfo.Frames.filter(box => box.FrameNr == Math.round(props.currentFrameNr ? props.currentFrameNr : 0)))
const predictedBoxesCurrentFrame = computed(() => props.predictedBoxes && props.predictedBoxes['boxes'] ? props.predictedBoxes['boxes'][props.currentFrameNr] : [])
const predictedBoxClassesCurrentFrame = computed(() => props.predictedBoxes && props.predictedBoxes['cls'] ? props.predictedBoxes['cls'][props.currentFrameNr] : [])

const boxesHovering = ref([])
const predictedBoxesHovering = ref([])

const boxColors = [
  '#bfdbfe',
  '#fef9c3',
  '#dc2626',
  '#fdba74',
  '#ec4899',
  '#bbf7d0',
  '#fee2e2',
  '#f3f4f6',
  '#67e8f9',
  '#ffffff',
  '#000000',
  '#555555',
  '#ee2299',
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

watch(() => props.drawPredictedBoxes, (n, o) => {
  resetCanvasAndDrawBoxes()
});

onMounted(async () => {

})
/* =========
** Functions
========= */

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

  // Draw predicted 
  if (props.drawPredictedBoxes) {
    Object.entries(predictedBoxesCurrentFrame.value).forEach(([idx, box]) => {
      // Received boxes are: array of 4 elements = array[xmin, ymin, xmax, ymax] but absolute values!
      let clsIdx = predictedBoxClassesCurrentFrame.value[idx]
      if (clsIdx < 2) {
        ctx.strokeStyle = boxColors[10 + clsIdx]
        const xleft = box[0] / props.videoinfo.Width * videoWidth.value
        const yleft = box[1] / props.videoinfo.Height * videoHeight.value
        const w = (box[2] - box[0]) / props.videoinfo.Width * videoWidth.value
        const h = (box[3] - box[1]) / props.videoinfo.Height * videoHeight.value
        ctx.strokeRect(xleft, yleft, w, h, 0.3);
      }
    })
  }

  // Draw labeled boxes
  Object.entries(boxes.value).forEach(([idx, box]) => {
    ctx.strokeStyle = boxColors[Number(box.LabelType)]
    const xleft = (box.X - box.Width / 2) * videoWidth.value
    const yleft = (box.Y - box.Height / 2) * videoHeight.value
    const w = box.Width * videoWidth.value
    const h = box.Height * videoHeight.value
    ctx.strokeRect(xleft, yleft, w, h);
  })

  // Draw team box
  let cfnr = String(Math.round(props.currentFrameNr ? props.currentFrameNr : 0))
  let teambox = props.videoinfo.TeamBoxes && Object.keys(props.videoinfo.TeamBoxes).includes(cfnr) ? props.videoinfo.TeamBoxes[cfnr] : null
  if (teambox) {
    ctx.strokeStyle = boxColors[12]
    const xleft = teambox['xmin'] * videoWidth.value
    const yleft = teambox['ymin'] * videoHeight.value
    const w = teambox['width'] * videoWidth.value
    const h = teambox['height'] * videoHeight.value
    ctx.strokeRect(xleft, yleft, w, h);
  }

  // Draw current drawing box
  if (!canvasmodeIsDelete.value && !canvasmodeIsAcceptPredictedBox.value) {
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

    predictedBoxesHovering.value = props.predictedBoxes?.boxes[props.currentFrameNr].filter(
      (boxArray) => {
        let minXbox = boxArray[0] / props.videoinfo.Width
        let maxXbox = boxArray[2] / props.videoinfo.Width
        let minYbox = boxArray[1] / props.videoinfo.Height
        let maxYbox = boxArray[3] / props.videoinfo.Height
        return minXbox < mouseX.value && mouseX.value < maxXbox && minYbox < mouseY.value && mouseY.value < maxYbox
      }
    )
  }
  mouse.value = (boxesHovering.value?.length || predictedBoxesHovering.value?.length) ? 'cursor-pointer' : canvasmodeIsDraw.value ? 'cursor-crosshair' : 'cursor-auto'
  
  resetCanvasAndDrawBoxes()
}

const canvasMouseLeave = (event) => {
  if (canvasmodeIsAcceptPredictedBox.value) { return }
  canvasMouseEndDrawing(event)
}

const canvasMouseEndDrawing = (event) => {
  mouse.value = ''
  mouseX.value = event.offsetX / videoWidth.value;
  mouseY.value = event.offsetY / videoHeight.value;

  if (canvasmodeIsDelete.value) {
    boxesHovering.value.forEach(box => emit("deleteBox", box))
  }

  if (canvasmodeIsAcceptPredictedBox.value) {
    if (predictedBoxesHovering.value.length < 2) {
      predictedBoxesHovering.value.forEach(coordinates => {
        let box = {
          "frameNr" : Math.round(props.currentFrameNr),
          "x" : (coordinates[2] + coordinates[0]) / 2 / props.videoinfo.Width,
          "y" : (coordinates[3] + coordinates[1]) / 2 / props.videoinfo.Height,
          "width" : (coordinates[2] - coordinates[0]) / props.videoinfo.Width,
          "height" : (coordinates[3] - coordinates[1]) / props.videoinfo.Height,
          "jumperVisible" : true,
        }
        emit('addBox', box)
      })
    }
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
    }

    if (box['height'] > 0.03) {
      emit('addBox', box)
    }

    mouseXstart.value = 0
    mouseYstart.value = 0
  }
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

