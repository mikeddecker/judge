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
        :current-frame-nr="currentFrame" :videoinfo="videoinfo" :labeltype="labeltypes[selectedLabeltype]"
        @play="updatePlaying" @pause="updatePaused" @seeked="onSeeked" @timeupdate="ontimeupdate"
        @add-box="onAddBox" @delete-box="onDeleteBox">
        </VideoPlayer>
        <SkillBalk :videoinfo="videoinfo" :Skills="skills" @skill-clicked="onSkillClicked" :currentFrame="currentFrame" class="mt-2"/>
        <SkillBalk :videoinfo="videoinfo" :Skills="predictions['skills']" @skill-clicked="onSkillClicked" :currentFrame="currentFrame" class="mt-2" :key="skillbalkKey"/>
        <div id="skill-controls" class="flex gap-2 my-2 wrap" v-show="modeIsSkills">
          <Button v-show="paused && !selectedSkill.Id" @click="setFrameStart()">set frame Start</Button>
          <Button v-show="paused && !selectedSkill.Id" @click="setFrameEnd()">set frame End</Button>
          <Button @click="playJustALittleFurther(-25)" class="bg-teal-600">-25</Button>
          <Button @click="playJustALittleFurther(-15)" class="bg-teal-600">-15</Button>
          <Button @click="playJustALittleFurther(-10)" class="bg-teal-600">-10</Button>
          <Button @click="playJustALittleFurther(-5)" class="bg-teal-600">-5</Button>
          <Button @click="playJustALittleFurther(-2)" class="bg-teal-600">-2</Button>
          <Button @click="playJustALittleFurther(-1)" class="bg-teal-600">-1</Button>
          <Button @click="playJustALittleFurther(+1)" class="bg-teal-600">+1</Button>
          <Button @click="playJustALittleFurther(+2)" class="bg-teal-600">+2</Button>
          <Button @click="playJustALittleFurther(+5)" class="bg-teal-600">+5</Button>
          <Button @click="playJustALittleFurther(+10)" class="bg-teal-600" ref="focusBtn">+10</Button>
          <Button @click="playJustALittleFurther(+15)" class="bg-teal-600">+15</Button>
          <Button @click="playJustALittleFurther(+25)" class="bg-teal-600">+25</Button>
          <Button v-show="selectedSkill.Id" @click="deselectSkill">Deselect skill</Button>
          <Button v-show="selectedSkill.Id && selectedSkill.FrameEnd != currentFrame" @click="frameToEndOfSkill">Frame to END of selected skill</Button>
          <Button v-show="selectedSkill.Id && selectedSkill.FrameEnd == currentFrame" @click="frameToStartOfSkill">Frame to START of selected skill</Button>
          <Button v-show="frameStart && frameEnd" @click="replaySection">Replay section</Button>
          <Button v-show="selectedSkill.Id" @click="playNextSection">Play next section</Button>
        </div>
        <div id="prediction-controls" class="flex gap-2 my-2 wrap" v-if="selectedSkillIsPrediction">
          <Button @click="splitPrediction">Split prediction</Button>
          <Button @click="mergeSplits">Merge</Button>
          <Button @click="() => shifPredictedSplitpoint(-1)">Shift splitpoint<i class="pi pi-arrow-left"></i></Button>
          <Button @click="() => shifPredictedSplitpoint(+1)">Shift splitpoint<i class="pi pi-arrow-right"></i></Button>
          <Button @click="acceptPredictedSkill">Accept<i class="pi pi-check"></i></Button>
        </div>
      </div>

      <div id="column-2" class="w-[25vw]">
        <div id="type-selection" class="flex h-fit gap-2 stretch">
          <Button :class="modeIsWatch ? 'p-button-highlight' : ''" @click="() => mode = 'WATCH'">Watch</Button>
          <Button :class="modeIsLocalize ? 'p-button-highlight' : ''" @click="() => mode = 'LOCALIZE'">Localize</Button>
          <Button :class="modeIsSegment ? 'p-button-highlight' : ''" @click="() => mode = 'SEGMENT'">Segment</Button>
          <Button :class="modeIsSkills ? 'p-button-highlight' : ''" @click="() => mode = 'SKILLS'">Skills</Button>
          <Button v-show="modeIsAnnotate" :class="modeIsAnnotate ? 'p-button-highlight' : ''" @click="() => predictMode = 'predict'">Annotate</Button>
          <Button v-show="modeIsPredict" :class="modeIsPredict ? 'p-button-highlight' : ''" @click="() => predictMode = 'annotate'">Predict</Button>

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
          <div class="flex gap-2">
            <span class="mr-2">Labeltype</span>
            <Select v-model="selectedLabeltype" :options="Object.keys(labeltypes)"></Select>
            <InputNumber v-model="currentFrame" inputId="input-currentFrame" fluid></InputNumber>
          </div>
          <div class="mt-2">
            <span class="mr-2">Canvas modus</span>
            <Select v-model="canvasMode" :options="canvasModes"></Select>
          </div>
          <div class="mt-2">
            <span class="mr-2">Use</span>
            <Select v-model="selectedModel" :options="modelOptions"></Select>
          </div>
        </div>
        <Button v-if="modeIsPredict" @click="() => predictSkills(videoinfo.Id)">Launch job</Button>
        
        <!--Skills -->
        <div id="skillinfo" v-if="modeIsSkills">
          <span>Start = {{ frameStart }}<br></span>
          <span>End = {{ frameEnd }}</span>
        </div>
        <div v-if="modeIsSkills" class="mx-2">
          <div v-for="(skillPropOptions, skillProp) in reversedSkillOptions" class="my-1">
            {{ skillProp }} <Select v-model="selectedSkill.ReversedSkillinfo[skillProp]" :options="Object.keys(skillPropOptions)" @update:model-value="updateLevel"></Select>
          </div>
          <Button v-show="frameStart && frameEnd && !selectedSkill.Id" @click="addSkill">Submit</Button>
          <Button v-show="selectedSkill.Id && !selectedSkillIsPrediction" @click="updateSkill">Update</Button>
          Level = {{ selectedSkillLevel }}
        </div>
      </div>
      
    </div>
    <pre>{{ videoinfo }}</pre>
    <Button v-if="modeIsSkills" class="mb-8" @click="toggleSkillsCompleted">Toggle skills completed, now = {{ videoinfo.Completed_Skill_Labels }}</button>
      
  </div>
  <div v-else>
    Loading...
  </div>
</template>

<script setup>
import SkillBalk from '@/components/SkillBalk.vue';
import VideoPlayer from '@/components/VideoPlayer.vue';
import { getVideoInfo, getVideoPath, getCroppedVideoPath, removeVideoFrame, postVideoFrame, getSkilloptions, postSkill, putSkill, getSkillLevel, updateVideoSkillsCompleted, getVideoPredictions, predictSkills } from '../services/videoService';
import { onMounted, ref, watch, computed, toRaw } from 'vue'
import { useRoute } from 'vue-router';
import LocalizeInfo from '@/components/LocalizeInfo.vue';

const route = useRoute()

// Loading
const loading = ref(false)
const error = ref('')
const croperror = ref('')

const videoId = ref(route.params.id)
const videoinfo = ref({})
const videoPath = ref('')
const croppedVideoSrc = ref('')
const paused = ref(true)
const videoElement = ref(null)
const skillbalkKey = ref(0)

const mode = ref('WATCH')
const modeIsWatch = computed(() => mode.value == 'WATCH')
const modeIsLocalize = computed(() => mode.value == 'LOCALIZE')
const modeIsSegment = computed(() => mode.value == 'SEGMENT')
const modeIsSkills = computed(() => mode.value == 'SKILLS')
const predictMode = ref('annotate')
const modeIsAnnotate = computed(() => predictMode.value == 'annotate')
const modeIsPredict = computed(() => predictMode.value == 'predict')

const canvasModes = ['draw', 'edit', 'delete', 'predict']
const labeltypes = {'individual' : 2, 'team': 1}
const selectedLabeltype = ref('individual')
const canvasMode = ref('draw')
const modelOptions = computed(() => ['yolov11n_ultralytics', 'yolov11n_run7'])
const selectedModel = ref('boxes')

const currentFrame = ref(0)
const frameStart = ref(currentFrame.value)
const frameEnd = ref(undefined)

const predictions = ref({'boxes': [], 'skills': []})
const skills = computed(() => {
  if (!videoinfo.value) { return [] }
  if (!videoinfo.value.Skills) { return [] }
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
const selectedSkill = ref({})
const selectedSkillLevel = ref(0)
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
const selectedSkillIsPrediction = computed(() => selectedSkill.value.hasOwnProperty('IsPrediction') && selectedSkill.value.IsPrediction) 


const sleep = ms => new Promise(r => setTimeout(r, ms));
watch(
  () => route.params.id,
  (newId) => (
    loadVideo(newId)
  )
)

function guidGenerator() {
    var S4 = function() {
       return (((1+Math.random())*0x10000)|0).toString(16).substring(1);
    };
    return (S4()+S4()+"-"+S4()+"-"+S4()+"-"+S4()+"-"+S4()+S4()+S4());
}

const getPreviousPredictedSkill = (adjacent) => {
  if (adjacent) {
    return predictions.value['skills'].filter(s => s.FrameEnd == selectedSkill.value.FrameStart)[0]
  } else {
    return predictions.value['skills'].filter(s => s.FrameEnd <= selectedSkill.value.FrameStart).reduce((acc, current) => acc ? (current.FrameEnd > acc.FrameEnd && current.FrameEnd <= selectedSkill.value.FrameStart ? current : acc) : current, null)
  }
}

const getNextPredictedSkill = (adjacent) => {
  if (adjacent) {
    return predictions.value['skills'].filter(s => s.FrameStart == selectedSkill.value.FrameEnd)[0]
  } else {
    return predictions.value['skills'].filter(s => s.FrameStart >= selectedSkill.value.FrameEnd).reduce((acc, current) => acc ? (current.FrameStart < acc.FrameStart && current.FrameStart >= selectedSkill.value.FrameEnd ? current : acc) : current, null)
  }
}

const updateLevel = async () => {
  if (frameStart.value) {
    let currentSkillinfo = normal2Reverse(selectedSkill.value.ReversedSkillinfo)
    let previousSkillinfo = null
    let previousSkillname = null
    if (selectedSkillIsPrediction.value) {
      let previousSkill = getPreviousPredictedSkill(true)
      if (previousSkill) {
        previousSkillinfo = previousSkill['Skillinfo']
        previousSkillname = previousSkill['ReversedSkillinfo']['Skill']
      }
    }
    selectedSkillLevel.value = await getSkillLevel(currentSkillinfo, previousSkillinfo, previousSkillname, frameStart.value, videoinfo.value.Id)
  }
}

const reverseDict = (d) => {
  return Object.fromEntries(Object.entries(d).map(([key, value]) => [value, key]))
}

const reverse2Normal = (rs) => {
  return Object.fromEntries(Object.entries(rs).map(([skillProp, reversedValue]) => [skillProp, skillOptions.value[skillProp][reversedValue] ? skillOptions.value[skillProp][reversedValue] : !!reversedValue ? "True" : "False"]))
}

const normal2Reverse = (ns) => {
  return Object.fromEntries(Object.entries(ns).map(([skillProp, reversedValue]) => [skillProp, reversedSkillOptions.value[skillProp][reversedValue]]))
}

const dictValueStringToInt = (d) => {
  return Object.fromEntries(Object.entries(d).map(([key, value]) => [key, Number(value)]))
}

onMounted(async () => {
  await loadVideo(videoId.value)
  videoElement.value = document.getElementById("vid")
})

async function loadVideo(id) {
  loading.value = true;
  try {
    videoPath.value = await getVideoPath(id)
    videoinfo.value = await getVideoInfo(id)
    let optionsLimbs = {"0": 0, "1": 1, "2": 2}
    let optionsBoolean = {"True": true, "False": false}
    reversedSkillOptions.value = {
      "Type" : await getSkilloptions("DoubleDutch", "Type").then(options => dictValueStringToInt(reverseDict(options))),
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
      "Turner1":  await getSkilloptions("DoubleDutch", "Turner").then(options => dictValueStringToInt(reverseDict(options))),
      "Turner2":  await getSkilloptions("DoubleDutch", "Turner").then(options => dictValueStringToInt(reverseDict(options))),
      "Skill" : await getSkilloptions("DoubleDutch", "Skill").then(options => dictValueStringToInt(reverseDict(options))), 
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

    videoinfo.value.Skills.forEach(s => {
      s["ReversedSkillinfo"] = reverse2Normal(s.Skillinfo)
    })
    
    selectedSkill.value["ReversedSkillinfo"] = Object.fromEntries(Object.entries(reversedSkillOptions.value).map(([skillprop, options]) => [skillprop, defaultOptions.value[skillprop]]))
    
    predictions.value = await getVideoPredictions(id).then(
      p => {
        p['skills'] = Object.entries(p['skills']).map(
          ([fs, pred]) => {
            let frameStart = Number(fs)
            let frameEnd = pred['Skill']['frameEnd']
            let skillproperties = Object.fromEntries(
              Object.entries(pred).map(
                ([skillprop, values]) => {
                  return [skillprop, values['y_pred']]
                }
              )
            )
            let transformedPrediction = {
              "Id": guidGenerator(),
              "IsPrediction" : true,
              "Skillinfo": skillproperties,
              "FrameStart": frameStart,
              "FrameEnd": frameEnd,
              "ReversedSkillinfo": reverse2Normal(skillproperties)
            }
            return transformedPrediction
          }
        )
        return p
      }
    )
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
  paused.value = false
}
function updatePaused(seconds) {
  currentFrame.value = Math.round(videoinfo.value.FPS * seconds)
  paused.value = true

}
function onSeeked(event) {
  frameStart.value = Math.round(event.target.off)
}
function ontimeupdate(seconds) {
  currentFrame.value = Math.round(videoinfo.value.FPS * seconds)
}

// Mode is localization
const setToNextFrame = () => {
  let minFrameNr = videoinfo.value.Frames
    .filter(b => b.LabelType == labeltypes[selectedLabeltype.value])
    .reduce((previous, current) => Math.min(previous, current.FrameNr), Infinity)
  let biggerFrameNr = videoinfo.value.Frames
    .filter(b => b.LabelType == labeltypes[selectedLabeltype.value])
    .filter((frameinfo) => frameinfo.FrameNr > currentFrame.value)
    .reduce((previous, current) => Math.min(previous, current.FrameNr), Infinity)
  currentFrame.value = biggerFrameNr == Infinity ? minFrameNr : biggerFrameNr

}

const setToPreviousFrame = () => {
  let maxFrameNr = videoinfo.value.Frames
    .filter(b => b.LabelType == labeltypes[selectedLabeltype.value])
    .reduce((previous, current) => Math.max(previous, current.FrameNr), -Infinity)
  let smallerFrameNr = videoinfo.value.Frames
    .filter((frameinfo) => frameinfo.FrameNr < currentFrame.value)
    .filter(b => b.LabelType == labeltypes[selectedLabeltype.value])
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
  currentFrame.value = rndFrameNr
}

const onAddBox = async (box) => {
  box['labeltype'] = labeltypes[selectedLabeltype.value]
  await postVideoFrame(videoinfo.value.Id, Math.round(currentFrame.value), box).then(vi => videoinfo.value = vi).catch(e => error.value = e)
}

const onDeleteBox = async (box) => {
  await removeVideoFrame(videoinfo.value.Id, Math.round(currentFrame.value), box).then(vi => videoinfo.value = vi).catch(err => error.value = err)
}

const setFrameStart = () => { frameStart.value = currentFrame.value }
const setFrameEnd = () => { frameEnd.value = currentFrame.value }

const play = () => {
  paused.value = false
  videoElement.value.play()
}

const onSkillClicked = (skillId, isPrediction) => {
  let skillsToFilter = isPrediction ? predictions.value['skills'] : skills.value

  let skill = skillsToFilter.filter(s => s.Id == skillId)[0]
  let skillinfo = skill['Skillinfo']
  skill['ReversedSkillinfo'] = reverse2Normal(skillinfo)
  
  selectedSkill.value = skill
  if (!paused.value) {
    videoElement.value.pause()
  }
  currentFrame.value = skill.FrameStart
  frameStart.value = skill.FrameStart
  frameEnd.value = skill.FrameEnd
  updateLevel()
}

async function playJustALittleFurther(framesToSkip) {
  if (!modeIsSkills.value) { return }
  if (framesToSkip < 0) {
    videoElement.value.currentTime += framesToSkip / videoinfo.value.FPS
  } else {
    let endTime = (currentFrame.value + framesToSkip) / videoinfo.value.FPS
    play()
    while (videoElement.value.currentTime < endTime) {
      await sleep(20)
    }
    videoElement.value.pause()
  }
  await sleep(270)
  if (frameStart.value && currentFrame.value != frameStart.value) {
    frameEnd.value = currentFrame.value
  }
}

function deselectSkill() {
  frameStart.value = frameEnd.value
  frameEnd.value = undefined
  if (!paused.value) {
    videoElement.value.pause()
  }
  currentFrame.value = frameStart.value
  selectedSkill.value = { "FrameStart": frameStart, "Skillinfo": normal2Reverse(defaultOptions.value), "ReversedSkillinfo": defaultOptions.value }
  videoElement.value.currentTime = frameStart.value / videoinfo.value.FPS
}

function frameToEndOfSkill() {
  currentFrame.value = selectedSkill.value.FrameEnd
}

function frameToStartOfSkill() {
  currentFrame.value = selectedSkill.value.FrameStart
}

async function replaySection() {
  currentFrame.value = frameStart.value
  await sleep(100)

  let endTime = frameEnd.value / videoinfo.value.FPS
  videoElement.value.play()
  while (videoElement.value.currentTime < endTime) {
    await sleep(10)
  }
  videoElement.value.pause()
  currentFrame.value = frameEnd.value
}
async function playNextSection() {
  let skills = selectedSkillIsPrediction.value ? predictions.value['skills'] : videoinfo.value.Skills
  let nextSkill = skills
    .filter(skill => skill.FrameStart >= selectedSkill.value.FrameEnd)
    .sort((a,b) => a.FrameEnd - b.FrameEnd)[0]
  if (nextSkill) {
    onSkillClicked(nextSkill.Id, selectedSkillIsPrediction.value)
    replaySection()
  }
}

async function addSkill() {
  let newSkill = {
    "FrameStart": frameStart.value,
    "FrameEnd" : frameEnd.value,
    "Skillinfo" : normal2Reverse(selectedSkill.value.ReversedSkillinfo)
  }
  videoinfo.value = await postSkill(videoinfo.value.Id, newSkill)
  prepareNextLabel(frameEnd.value)
}

function prepareNextLabel(fs) {
  frameStart.value = fs
  frameEnd.value = undefined
  for (let skillIdx in skills.value) {
    skills.value[skillIdx].inCreation = false
  }
}

async function updateSkill() {
  let copy = structuredClone(toRaw(selectedSkill.value))
  copy.Skillinfo = normal2Reverse(selectedSkill.value.ReversedSkillinfo)
  videoinfo.value = await putSkill(videoinfo.value.Id, copy)
  deselectSkill()
  prepareNextLabel(copy.FrameEnd)
}

async function toggleSkillsCompleted() {
  updateVideoSkillsCompleted(videoinfo.value.Id, !videoinfo.value.Completed_Skill_Labels).then(() => videoinfo.value.Completed_Skill_Labels = ! videoinfo.value.Completed_Skill_Labels)
}

function shifPredictedSplitpoint(addFrames) {
  if (!selectedSkillIsPrediction.value) { return }
  if (currentFrame.value != selectedSkill.value.FrameStart && currentFrame.value != selectedSkill.value.FrameEnd) { return }
  
  let shiftedFrameNr = Math.round(currentFrame.value + addFrames)
  if (currentFrame.value == selectedSkill.value.FrameEnd) {
    let nexSkill = getNextPredictedSkill(true)
    nexSkill.FrameStart = shiftedFrameNr
    selectedSkill.value.FrameEnd = shiftedFrameNr
    frameEnd.value = shiftedFrameNr
  }

  if (currentFrame.value == selectedSkill.value.FrameStart) {
    let previousSkill = getPreviousPredictedSkill(true)
    previousSkill.FrameEnd = shiftedFrameNr
    selectedSkill.value.FrameStart = shiftedFrameNr
    frameStart.value = shiftedFrameNr
  }
  currentFrame.value = shiftedFrameNr
}

const acceptPredictedSkill = async () => {
  selectedSkill.value.Skillinfo = normal2Reverse(selectedSkill.value.ReversedSkillinfo)
  videoinfo.value = await postSkill(videoinfo.value.Id, {
    'FrameStart' : selectedSkill.value.FrameStart,
    'FrameEnd' : selectedSkill.value.FrameEnd,
    'Skillinfo' : selectedSkill.value.Skillinfo
  })
  prepareNextLabel(frameEnd.value)
  playNextSection()
}

const mergeSplits = () => {

  if (currentFrame.value == selectedSkill.value.FrameEnd) {
    let nexSkill = getNextPredictedSkill(true)
    let nextFrameEnd = nexSkill.FrameEnd
    let idx = predictions.value["skills"].indexOf(nexSkill)
    predictions.value["skills"].splice(idx, 1)

    selectedSkill.value.FrameEnd = nextFrameEnd
    frameEnd.value = nextFrameEnd
    currentFrame.value = nextFrameEnd
  }

  if (currentFrame.value == selectedSkill.value.FrameStart) {
    let previousSkill = getPreviousPredictedSkill(true)
    let previsousFrameStart = previousSkill.FrameStart
    let idx = predictions.value["skills"].indexOf(previousSkill)
    predictions.value["skills"].splice(idx, 1)

    selectedSkill.value.FrameStart = previsousFrameStart
    frameStart.value = previsousFrameStart
    currentFrame.value = previsousFrameStart
  }
}

const splitPrediction = async () => {
  let copy = structuredClone(toRaw(selectedSkill.value))
  let splitpoint = Math.round((copy.FrameEnd + copy.FrameStart) / 2)
  
  selectedSkill.value.FrameEnd = splitpoint
  frameEnd.value = splitpoint
  copy.FrameStart = splitpoint
  copy.Id = `${copy.Id}${splitpoint}`
  predictions.value["skills"].push(copy)
  currentFrame.value = splitpoint
  skillbalkKey.value += 1
}

</script>

<style scoped>
.p-button-highlight {
  background-color: var(--p-button-primary-active-background);
}

.error {
  color: red;
}

button:focus {
  border-color: blue;
  outline: none;
}
</style>
