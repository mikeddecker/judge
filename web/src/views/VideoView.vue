<template>
  <div v-if="videoinfo">
    <h1>{{ videoinfo.Name }}</h1>
    <div v-if="loading">Loading...</div>
    <span v-if="error" class="error">{{ error }}</span>
    <span v-if="croperror">{{ croperror }}</span>
    
    <div id="videoview-content" v-if="!loading" class="flex gap-2">
      <div id="column-1" class="w-[75vw]">

        <VideoPlayer class=" relative" 
        v-if="!loading" v-bind:video-id="route.params.id" :video-src="videoPath" :mode="mode" :canvas-mode="canvasMode"
        :current-frame-nr="currentFrame" :videoinfo="videoinfo"
        @play="updatePlaying" @pause="updatePaused" @seeked="onSeeked" @timeupdate="ontimeupdate"
        @add-box="onAddBox" @delete-box="onDeleteBox">
        </VideoPlayer>
        <SkillBalk :videoinfo="videoinfo" :Skills="skills" @skill-clicked="onSkillClicked" :currentFrame="currentFrame" class="mt-2"/>
      </div>

      <div id="column-2" class="w-[25vw]">
        <div id="type-selection" class="flex h-fit gap-2 stretch">
          <Button :class="modeIsWatch ? 'p-button-highlight' : ''" @click="() => mode = 'WATCH'">Watch</Button>
          <Button :class="modeIsLocalize ? 'p-button-highlight' : ''" @click="() => mode = 'LOCALIZE'">Localize</Button>
          <Button :class="modeIsSegment ? 'p-button-highlight' : ''" @click="() => mode = 'SEGMENT'">Segment</Button>
          <Button :class="modeIsSkills ? 'p-button-highlight' : ''" @click="() => mode = 'SKILLS'">Skills</Button>
        </div>
        <div class="my-2 flex gap-2">
          currentFrame: {{ currentFrame }}
          <div id="localize-frame-navigation-buttons" class="flex gap-2">
            <Button v-if="modeIsLocalize" @click="setToPreviousFrame"><i class="pi pi-arrow-left"></i></Button>
            <Button v-if="modeIsLocalize" @click="setToRandomFrame"><i class="pi pi-arrow-right-arrow-left"></i></Button>
            <Button v-if="modeIsLocalize" @click="setToNextFrame"><i class="pi pi-arrow-right"></i></Button>
          </div>
        </div>
        <LocalizeInfo v-if="modeIsLocalize" :videoinfo="videoinfo"></LocalizeInfo>
        <div id="localize-controls" v-if="modeIsLocalize" class="my-2">
          <div>
            <span class="mr-2">Canvas modus</span>
            <Select v-model="canvasMode" :options="canvasModes"></Select>
          </div>
          <div class="mt-2">
            <span class="mr-2">Use</span>
            <Select v-model="selectedModel" :options="modelOptions"></Select>
          </div>
        </div>

        <!--Skills -->
        <div v-if="modeIsSkills" class="mx-2">
          <div v-for="(skillPropOptions, skillProp) in reversedSkillOptions" class="my-1">
            {{ skillProp }} <Select v-model="selectedSkillinfo[skillProp]" :options="Object.keys(skillPropOptions)"></Select>
          </div>
          {{ selectedSkillinfo }}
        </div>
      </div>
      
    </div>
      <pre>{{ videoinfo }}</pre>
      
  </div>
  <div v-else>
    Loading...
  </div>
</template>

<script setup>
import SkillBalk from '@/components/SkillBalk.vue';
import VideoPlayer from '@/components/VideoPlayer.vue';
import { getVideoInfo, getVideoPath, getCroppedVideoPath, removeVideoFrame, postVideoFrame, getSkilloptions } from '../services/videoService';
import { onMounted, ref, watch, computed } from 'vue'
import { useRoute } from 'vue-router';
import LocalizeInfo from '@/components/LocalizeInfo.vue';
import LocalizeControls from '@/components/LocalizeControls.vue';

const route = useRoute()

// Loading
const loading = ref(false)
const error = ref('')
const croperror = ref('')

const videoId = ref(route.params.id)
const videoinfo = ref({})
const videoPath = ref('')
const croppedVideoSrc = ref('')

const mode = ref('WATCH')
const modeIsWatch = computed(() => mode.value == 'WATCH')
const modeIsLocalize = computed(() => mode.value == 'LOCALIZE')
const modeIsSegment = computed(() => mode.value == 'SEGMENT')
const modeIsSkills = computed(() => mode.value == 'SKILLS')

const canvasModes = ['draw', 'edit', 'delete']
const canvasMode = ref('draw')
const modelOptions = ['boxes', 'yolov11n_ultralytics', 'yolov11n_run7']
const selectedModel = ref('boxes')

const currentFrame = ref(0)
const frameStart = ref(currentFrame.value)
const frameEnd = ref(undefined)

const skills = computed(() => {
  if (!videoinfo.value) { return [] }
  if (!videoinfo.value.Skills) { return [] }
  console.log(videoinfo.value.Skills)
  let s = videoinfo.value ? [...videoinfo.value.Skills] : []
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

const skillOptions = ref({})
const reversedSkillOptions = ref({})
const selectedSkillinfo = ref({})
const defaultOptions = ref({
  "Type" : "Double Dutch",
  "Rotations" : "1 rotation",
  "Turner1":  "normal",
  "Turner2":  "normal",
  "Skill" : "jump",
  "Hands" : "0",
  "Feet" : "2",
  "Turntable" : "0 roations",
  "BodyRotations" : "0 roations",
  "Backwards" : "False",
  "Sloppy" : "False",
  "Hard2see" : "False",
  "Fault" : "False",
})


const sleep = ms => new Promise(r => setTimeout(r, ms));
watch(
  () => route.params.id,
  (newId) => (
    loadVideo(newId)
  )
)

const reverseDict = (d) => {
  return Object.fromEntries(Object.entries(d).map(([key, value]) => [value, key]))
}

const reverse2Normal = (rs) => {
  return Object.fromEntries(Object.entries(rs).map(([skillProp, reversedValue]) => [skillProp, skillOptions.value[skillProp][reversedValue]]))
}

const normal2Reverse = (ns) => {
  return Object.fromEntries(Object.entries(ns).map(([skillProp, reversedValue]) => [skillProp, reversedSkillOptions.value[skillProp][reversedValue]]))
}

onMounted(async () => {
  await loadVideo(videoId.value)
})
async function loadVideo(id) {
  loading.value = true;
  try {
    videoPath.value = await getVideoPath(id)
    videoinfo.value = await getVideoInfo(id)
    let optionsLimbs = {"0": 0, "1": 1, "2": 2}
    let optionsBoolean = {"True": true, "False": false}
    reversedSkillOptions.value = {
      "Type" : await getSkilloptions("DoubleDutch", "Type").then(options => reverseDict(options)),
      "Rotations" : {
        "0 roations" : 0,
        "1 rotation" : 1,
        "2 rotation" : 2,
        "3 rotations" : 3,
        "4 rotations" : 4,
        "5 rotations" : 5,
        "6 rotations" : 6,
        "7 rotations" : 7,
        "8 rotations" : 8,
      }, 
      "Turner1":  await getSkilloptions("DoubleDutch", "Turner").then(options => reverseDict(options)),
      "Turner2":  await getSkilloptions("DoubleDutch", "Turner").then(options => reverseDict(options)),
      "Skill" : await getSkilloptions("DoubleDutch", "Skill").then(options => reverseDict(options)), 
      "Hands" : optionsLimbs,
      "Feet" : optionsLimbs, 
      "Turntable" : {
        "0 roations"     : 0,
        "0.25 rotations" : 1,
        "0.50 rotations" : 2,
        "0.75 rotations" : 3,
        "1 rotations"    : 4,
      }, 
      "BodyRotations" : {
        "0 roations" : 0,
        "0.5 rotations" : 1,
        "1 rotation" : 2,
      }, 
      "Backwards" : optionsBoolean, 
      "Sloppy" : optionsBoolean, 
      "Hard2see" : optionsBoolean, 
      "Fault" : optionsBoolean, 
    }

    optionsLimbs = {0: "0", 1: "1", 2: "2"}
    optionsBoolean = { true: "True", false: "False"}
    skillOptions.value = {
      "Type" : await getSkilloptions("DoubleDutch", "Type"),
      "Rotations" : {
        0: "0 roations",
        1: "1 rotation",
        2: "2 rotation",
        3: "3 rotations",
        4: "4 rotations",
        5: "5 rotations",
        6: "6 rotations",
        7: "7 rotations",
        8: "8 rotations",
      }, 
      "Turner2":  await getSkilloptions("DoubleDutch", "Turner"),
      "Turner1":  await getSkilloptions("DoubleDutch", "Turner"),
      "Skill" : await getSkilloptions("DoubleDutch", "Skill"), 
      "Hands" : optionsLimbs,
      "Feet" : optionsLimbs, 
      "Turntable" : {
        0: "0 roations"    ,
        1: "0.25 rotations",
        2: "0.50 rotations",
        3: "0.75 rotations",
        4: "1 rotations"   ,
      }, 
      "BodyRotations" : {
        0 : "0 roations",
        1 : "0.5 rotations",
        2 : "1 rotation",
      }, 
      "Backwards" : optionsBoolean, 
      "Sloppy" : optionsBoolean, 
      "Hard2see" : optionsBoolean, 
      "Fault" : optionsBoolean, 
    }
    
    selectedSkillinfo.value = Object.fromEntries(Object.entries(reversedSkillOptions.value).map(([skillprop, options]) => [skillprop, defaultOptions.value[skillprop]]))
    console.log(defaultOptions.value)
  } catch (e) {
    console.error(e)
    error.value = 'Failed To load';
  } finally {
    loading.value = false;
  }

  try {
    croppedVideoSrc.value = await getCroppedVideoPath(id)
  } catch {
    croperror.value = 'No cropped video available'
  }
}

function updatePlaying(event) {
  console.log("updatePlaying", event)
}
function updatePaused(seconds) {
  console.log("updatePaused", seconds)
  currentFrame.value = Math.round(videoinfo.value.FPS * seconds)
  //   croppedVideoElement.value.currentTime = videoElement.value.currentTime

}
function onSeeked(event) {
  console.log("onSeeked", event)
}
function ontimeupdate(seconds) {
  currentFrame.value = Math.round(videoinfo.value.FPS * seconds)
}

// Mode is localization
const setToNextFrame = () => {
  console.log(currentFrame.value)
  let minFrameNr = videoinfo.value.Frames
    .filter(b => b.LabelType == 2)
    .reduce((previous, current) => Math.min(previous, current.FrameNr), Infinity)
  let biggerFrameNr = videoinfo.value.Frames
    .filter(b => b.LabelType == 2)
    .filter((frameinfo) => frameinfo.FrameNr > currentFrame.value)
    .reduce((previous, current) => Math.min(previous, current.FrameNr), Infinity)
  currentFrame.value = biggerFrameNr == Infinity ? minFrameNr : biggerFrameNr

}

const setToPreviousFrame = () => {
  let maxFrameNr = videoinfo.value.Frames
    .filter(b => b.LabelType == 2)
    .reduce((previous, current) => Math.max(previous, current.FrameNr), -Infinity)
  let smallerFrameNr = videoinfo.value.Frames
    .filter((frameinfo) => frameinfo.FrameNr < currentFrame.value)
    .filter(b => b.LabelType == 2)
    .reduce((previous, current) => Math.max(previous, current.FrameNr), -Infinity)
  currentFrame.value = smallerFrameNr == -Infinity ? maxFrameNr : smallerFrameNr
}

const setToRandomFrame = () => {
  let rndTime = 0
  let rndFrameNr = 0
  let frameNrAlreadyLabeled = true
  while (frameNrAlreadyLabeled) {
    rndTime = Math.random() * videoinfo.value.Duration
    rndFrameNr = Math.floor(rndTime * videoinfo.value.FPS)
    frameNrAlreadyLabeled = videoinfo.value.Frames.map(frameinfo => frameinfo.FrameNr).includes(rndFrameNr)
  }
  console.log("chae", rndFrameNr)
  currentFrame.value = rndFrameNr
}

const onAddBox = async (box) => {
  await postVideoFrame(videoinfo.value.Id, Math.round(currentFrame.value), box).then(vi => videoinfo.value = vi).catch(e => error.value = e)
}

const onDeleteBox = async (box) => {
  await removeVideoFrame(videoinfo.value.Id, Math.round(currentFrame.value), box).then(vi => videoinfo.value = vi).catch(err => error.value = err)
}


const onSkillClicked = (skillId) => {
  let skill = skills.value.filter(s => s.Id == skillId)[0]
  let skillinfo = skill['Skillinfo']
  console.log('onSkillClicked', skillinfo)
  console.log('re-reversed', reverse2Normal(skillinfo))
  console.log('normal', normal2Reverse(reverse2Normal(skillinfo)))
}
</script>

<style scoped>
.p-button-highlight {
  background-color: var(--p-button-primary-active-background);
}

.error {
  color: red;
}
</style>
