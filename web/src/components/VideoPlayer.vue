<template>
  <p>LabeledFrames: {{ labeledFramesCount }} | Current frame : {{ currentFrame }} | FramesLabeledPerSecond : {{ framesLabeledPerSecond }} | total labels: {{ totalLabels }} | Full box for all jumpers : {{ modeLocalizationIsAll }}</p>
  <div class="container">
    <video class="absolute"
    id="vid"
    ref="videoPlayer" :src="videoSrc"
    controls autoplay loop 
    @canplay="updatePaused" @playing="updatePaused" @pause="updatePaused"
    >
    </video>
    <canvas
      v-show="modeIsReview | modeIsLocalization"
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
    <SkillBalk v-show="modeIsSkills"
      :videoinfo="vidinfo" 
      :Skills="skills" 
      @skill-clicked="onSkillClicked"
      :currentFrame="currentFrame"
    />
    <div class="controls">
      <button @click="toggleLabelMode">Current modus: {{ labelMode }}</button>
      <button v-show="modeLocalizationIsAll && !modeIsSkills" @click="toggleLocalizationType">1 box 4 all</button>
      <button v-show="!modeLocalizationIsAll && !modeIsSkills" @click="toggleLocalizationType">1 box / jumper</button>
      <button v-show="paused && modeIsLocalization" @click="play">&#9654;</button>
      <button v-show="playing" @click="pause">&#9208;</button>
      <button v-show="modeIsLocalization && modeLocalizationIsAll" @click="postFullFrameLabelAndDisplayNextFrame">label as full screen</button>
      <button v-show="modeIsLocalization" @click="displayNextRandomFrame">random next frame</button>
      <button v-show="modeIsSkills && paused" @click="setFrameStartEnd('start')">set frameStart</button>
      <button v-show="modeIsSkills && paused" @click="setFrameStartEnd('end')">set frameEnd</button>
      <button v-show="modeIsSkills" @click="playJustALittleFurther(-25)">-25</button>
      <button v-show="modeIsSkills" @click="playJustALittleFurther(-15)">-15</button>
      <button v-show="modeIsSkills" @click="playJustALittleFurther(-10)">-10</button>
      <button v-show="modeIsSkills" @click="playJustALittleFurther(-5)">-5</button>
      <button v-show="modeIsSkills" @click="playJustALittleFurther(-2)">-2</button>
      <button v-show="modeIsSkills" @click="playJustALittleFurther(-1)">-1</button>
      <button v-show="modeIsSkills" @click="playJustALittleFurther(+1)">+1</button>
      <button v-show="modeIsSkills" @click="playJustALittleFurther(+2)">+2</button>
      <button v-show="modeIsSkills" @click="playJustALittleFurther(+5)">+5</button>
      <button v-show="modeIsSkills" @click="playJustALittleFurther(+10)">+10</button>
      <button v-show="modeIsSkills" @click="playJustALittleFurther(+15)">+15</button>
      <button v-show="modeIsSkills" @click="playJustALittleFurther(+25)">+25</button>
      <button v-show="modeIsSkills && selectedSkill" @click="deselectSkill()">Deselect skill</button>
      <button v-show="modeIsSkills && selectedSkill" @click="frameToEndOfSkill()">Frame to end of selected skill</button>
      <div class="review-controls" v-show="modeIsReview">
        <button class="big-arrow" @click="setToPreviousFrameAndDraw">&larr;</button>
        <button class="big-arrow" @click="setToNextFrameAndDraw">&rarr;</button>
        <button v-show="modeIsReview && modeLocalizationIsAll" @click="deleteLabel"><img src="@/assets/delete.png" alt="buttonpng" class="icon"/></button>
      </div>
      <BoxCard v-for="(box, index) in currentBoxes" :key="index" :frameinfo="box" @deleteBox="deleteLabel"/>
    </div>
    <p v-if="selectedSkill" style="width: 100%;">Selected skill: {{ selectedSkill }}</p>
    <p>FrameStart = {{ frameStart }} | FrameEnd = {{ frameEnd }} | skill can update {{ skillCanUpdate }}</p>
    <div v-if="optionsLoaded && modeIsSkills" class="controls">
      <SelectComponent :skilltype="'Type'" :options="optionsType" title="Type" :defaultValue="selectedSkill ? selectedSkill.Skillinfo['Type'] : selectedOptions['Type'][0]" @update:selected="handleSelectedChange"/>
      <SelectComponent :skilltype="'Rotations'" :options="optionsRotations" title="Rotations" :defaultValue="selectedSkill ? selectedSkill.Skillinfo['Rotations'] : selectedOptions['Rotations'][0]" @update:selected="handleSelectedChange"/>
      <SelectComponent :skilltype="'Turner1'" :options="optionsTurner" title="Turner1" :defaultValue="selectedSkill ? selectedSkill.Skillinfo['Turner1'] : selectedOptions['Turner1'][0]" @update:selected="handleSelectedChange"/>
      <SelectComponent :skilltype="'Turner2'" :options="optionsTurner" title="Turner2" :defaultValue="selectedSkill ? selectedSkill.Skillinfo['Turner2'] : selectedOptions['Turner2'][0]" @update:selected="handleSelectedChange"/>
      <SelectComponent :skilltype="'Skill'" :options="optionsSkill" title="Skill" :defaultValue="selectedSkill ? selectedSkill.Skillinfo['Skill'] : selectedOptions['Skill'][0]" @update:selected="handleSelectedChange"/>
      <SelectComponent :skilltype="'Hands'" :options="optionsLimbs" title="Hands" :defaultValue="selectedSkill ? selectedSkill.Skillinfo['Hands'] : selectedOptions['Hands'][0]" @update:selected="handleSelectedChange"/>
      <SelectComponent :skilltype="'Feet'" :options="optionsLimbs" title="Feet" :defaultValue="selectedSkill ? selectedSkill.Skillinfo['Feet'] : selectedOptions['Feet'][0]" @update:selected="handleSelectedChange"/>
      <SelectComponent :skilltype="'Turntable'" :options="optionsTurntable" title="Turntable" :defaultValue="selectedSkill ? selectedSkill.Skillinfo['Turntable'] : selectedOptions['Turntable'][0]" @update:selected="handleSelectedChange"/>
      <SelectComponent :skilltype="'BodyRotations'" :options="optionsBodyRotations" title="BodyRotations" :defaultValue="selectedSkill ? selectedSkill.Skillinfo['BodyRotations'] : selectedOptions['BodyRotations'][0]" @update:selected="handleSelectedChange"/>
      <SelectComponent :skilltype="'Backwards'" :options="optionsBoolean" title="Backwards" :defaultValue="selectedSkill ? selectedSkill.Skillinfo['Backwards'] : selectedOptions['Backwards'][0]" @update:selected="handleSelectedChange"/>
      <SelectComponent :skilltype="'Sloppy'" :options="optionsBoolean" title="Sloppy" :defaultValue="selectedSkill ? selectedSkill.Skillinfo['Sloppy'] : selectedOptions['Sloppy'][0]" @update:selected="handleSelectedChange"/>
      <button v-show="frameStart && frameEnd && !selectedSkill" @click="addSkill">submit skill</button>
      <button v-show="skillCanUpdate" @click="updateSkill">update</button>
      <button v-if="selectedSkill" @click="removeSkill"><img src="@/assets/delete.png" alt="buttonpng" class="icon"/></button>
    </div>
  </div>
  <pre>{{ selectedOptions }}</pre>
  <pre>{{ vidinfo }}</pre>
</template>

<script setup>
import { deleteSkill, getSkilloptions, getVideoInfo, postSkill, postVideoFrame, putSkill, removeVideoFrame } from '@/services/videoService';
import { computed, onBeforeMount, onMounted, ref } from 'vue';
import { useRouter } from 'vue-router'
import BoxCard from './BoxCard.vue';
import SkillBalk from './SkillBalk.vue';
import SelectComponent from './SelectComponent.vue';

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

const skills = computed(() => {
  let s = vidinfo.value ? [...vidinfo.value.Skills] : []
  if (frameStart.value && currentFrame.value >= frameStart.value) {
    let skillInCreation = {
      "Id" : 0,
      "inCreation" : true,
      "FrameStart": frameStart.value,
      "FrameEnd": frameEnd.value ? frameEnd.value : currentFrame.value,
    }
    s.push(skillInCreation)
  }
  return s
})
const selectedSkill = ref(undefined)
const frameStart = ref(undefined)
const frameEnd = ref(undefined)
const optionsType = ref([])
const optionsTurner = ref([])
const optionsSkill = ref([])
const optionsRotations = {
  0 : "0 roations",
  1 : "1 rotation",
  2 : "2 rotation",
  3 : "3 rotations",
  4 : "4 rotations",
  5 : "5 rotations",
  6 : "6 rotations",
  7 : "7 rotations",
  8 : "8 rotations",
}
const optionsLimbs = {
  0 : 0,
  1 : 1,
  2 : 2,
}
const optionsTurntable = {
  0 : "0 roations",
  1 : "0.25 rotations",
  2 : "0.50 rotations",
  3 : "0.75 rotations",
  4 : "1 rotations",
  5 : "1.25 rotations",
  6 : "1.50 rotations",
}
const optionsBodyRotations = {
  0 : "0 roations",
  1 : "1 rotation",
  2 : "2 rotation",
}
const optionsBoolean = {true:true, false:false}
const selectedOptions = ref({})
const optionsLoaded = ref(false)
const skillCanUpdate = ref(false)

const labeledFramesCount = computed(() => vidinfo.value ? vidinfo.value.LabeledFrameCount : 0)
const labelMode = ref("skills")
const modeIsLocalization = computed(() => { return labelMode.value == "localization" })
const modeIsSkills = computed(() => { return labelMode.value == "skills" })
const modeIsReview = computed(() => { return labelMode.value == "review" })
const modeLocalizationIsAll = ref(false)
const currentFrameIdx = ref(0)
const framesLabeledPerSecond = computed(() => { return vidinfo.value ? vidinfo.value.FramesLabeledPerSecond.toFixed(2) : 0 })
const totalLabels = ref(0)
const avgLabels = ref(0)
const currentBoxes = ref([])
const sleep = ms => new Promise(r => setTimeout(r, ms));

onBeforeMount(async () => {
    vidinfo.value = await getVideoInfo(props.videoId);
    optionsType.value = await getSkilloptions("DoubleDutch", "Type")
    optionsTurner.value = await getSkilloptions("DoubleDutch", "Turner")
    optionsSkill.value = await getSkilloptions("DoubleDutch", "Skill")
    setDefaultSelectedOptions()
    optionsLoaded.value = true
})
onMounted(async () => {
  videoElement.value = document.getElementById("vid")
})

function updatePaused(event) {
  videoduration.value = event.target.duration
  currentWidth.value = event.target.clientWidth
  currentHeight.value = event.target.clientHeight
  if (!paused.value == true) {
    currentFrame.value = Math.floor(!modeIsReview.value ? vidinfo.value.FPS * event.target.currentTime : vidinfo.value.Frames[currentFrameIdx.value].FrameNr)
  }
  paused.value = event.target.paused;
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
  if (frameinfo['height'] > 0.03) {
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
  if (Math.random() < (framesLabeledPerSecond.value - avgLabels.value) / 100 && modeLocalizationIsAll.value) {
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
    labelMode.value = "skills"
    pause()
    setCurrentTime(currentFrame.value / vidinfo.value.FPS)
  } else if (modeIsSkills.value) {
    if (vidinfo.value.Frames.length > 0) {
      labelMode.value = "review"
      pause()
      currentFrame.value = vidinfo.value.Frames[0].FrameNr
      setCurrentTime(currentFrame.value / vidinfo.value.FPS)
      drawCurrentFrame()
    } else {
      labelMode.value = "localization"
      currentBoxes.value = []
      clearAndReturnCtx()
    }
  } else if (modeIsReview.value) {
    labelMode.value = "localization"
    currentBoxes.value = []
    clearAndReturnCtx()
  }
}
function toggleLocalizationType() {
  modeLocalizationIsAll.value = !modeLocalizationIsAll.value
}
function setFrameStartEnd(start_or_end) {
  if (start_or_end == 'start') {
    if (selectedSkill.value && frameStart.value != currentFrame.value) {
      skillCanUpdate.value = true
    }
    frameStart.value = currentFrame.value
    if (selectedSkill.value) { selectedSkill.value.FrameStart = currentFrame.value }
  } else {
    if (frameStart.value == undefined || frameStart.value > currentFrame.value) { return }
    if (selectedSkill.value && frameEnd.value != currentFrame.value) {
      skillCanUpdate.value = true
      selectedSkill.value.FrameEnd = currentFrame.value
    }
    if (frameStart.value >= currentFrame.value) { 
      return 
    } else {
      frameEnd.value = currentFrame.value
    }
  }
}

//////////////////////////////////////////////////////////////////////////////
// Skils
//////////////////////////////////////////////////////////////////////////////
async function playJustALittleFurther(framesToSkip) {
  if (!modeIsSkills.value) { return }
  // setCurrentTime(currentFrame.value / vidinfo.value.FPS)
  if (framesToSkip < 0) {
    videoElement.value.currentTime += framesToSkip / vidinfo.value.FPS
    currentFrame.value = Math.round(vidinfo.value.FPS * videoElement.value.currentTime)
  } else {
    let timeToSkip = 1 / vidinfo.value.FPS * framesToSkip * 1000
    videoElement.value.play()
    await sleep(timeToSkip)
    videoElement.value.pause()
  }
}
function onSkillClicked(skillIdentifier) {
  let isClicked = selectedSkill.value ? selectedSkill.value.Id == skillIdentifier : false
  for (let skillIdx in skills.value) {
    skills.value[skillIdx].inCreation = false
  }
  selectedSkill.value = isClicked ? undefined : skills.value.filter(s => s.Id == skillIdentifier)[0]
  if (!isClicked) {
    selectedSkill.value.inCreation = !isClicked
  } 
  if (selectedSkill.value) {
    frameStart.value = selectedSkill.value.FrameStart
    frameEnd.value = selectedSkill.value.FrameEnd
    currentFrame.value =  selectedSkill.value.FrameStart
    setCurrentTime(selectedSkill.value.FrameStart / vidinfo.value.FPS)
  } else {
    frameStart.value = frameEnd.value = undefined
    skillCanUpdate.value = false
  }
}
function setDefaultSelectedOptions(fs) {
  selectedOptions.value = {
    "Type" : [1, "DoubleDutch"],
    "Rotations" : [1, 1],
    "Turner1": [1, "/"],
    "Turner2": [1, "/"],
    "Skill" : [1, "jump"],
    "Hands" : [0, 0],
    "Feet" : [2, 2],
    "Turntable" : [0, 0],
    "BodyRotations" : [0, 0],
    "Backwards" : [false, false],
    "Sloppy" : [false, false],
  }
  selectedSkill.value = undefined
  skillCanUpdate.value = false
  frameStart.value = fs
  frameEnd.value = undefined
  for (let skillIdx in skills.value) {
    skills.value[skillIdx].inCreation = false
  }
}
function handleSelectedChange(skillinfo, value, description) {
  selectedOptions.value[skillinfo] = [Number(value) ? Number(value) : Boolean(value), description]
  if (selectedSkill.value) {
    selectedSkill.value["Skillinfo"][skillinfo] = Number(value) ? Number(value) : Boolean(value)
    skillCanUpdate.value = true
  }
}
async function addSkill() {
  let newSkill = {
    "frameStart": frameStart.value,
    "frameEnd" : frameEnd.value,
    "skillinfo" : Object.entries(selectedOptions.value).reduce((newDict, [key, value]) => {
      newDict[key] = value[0];
      return newDict;
    }, {})
  }
  vidinfo.value = await postSkill(vidinfo.value.Id, newSkill)
  setDefaultSelectedOptions(frameEnd.value)
}
async function updateSkill() {
  let updatedSkill = Object.entries(selectedSkill.value).reduce((newDict, [key, value]) => {
    newDict[key] = value;
    return newDict;
  }, {})
  const updatedVideoinfo = await putSkill(vidinfo.value.Id, updatedSkill)
  vidinfo.value = updatedVideoinfo
  skillCanUpdate.value = false
  setDefaultSelectedOptions()
}
async function removeSkill() {
  vidinfo.value = await deleteSkill(vidinfo.value.Id, selectedSkill.value.FrameStart, selectedSkill.value.FrameEnd)
  skillCanUpdate.value = false
  setDefaultSelectedOptions()
}
function deselectSkill() {
  skillCanUpdate.value = false
  frameStart.value = frameEnd.value
  frameEnd.value = undefined
  for (let skillIdx in skills.value) {
    skills.value[skillIdx].inCreation = false
  }
  selectedSkill.value = undefined
}
function frameToEndOfSkill() {
  currentFrame.value = selectedSkill.value.FrameEnd
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
  max-width: 95%;
  max-height: 80vh;
}

.controls {
  margin-top: 0.3rem;
  display: flex;
  flex-wrap: wrap;
}

.review-controls {
  margin: 0.2rem 0;
}

button {
  min-width: 4rem;
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
}
</style>
