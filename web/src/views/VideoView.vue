<template>
  <div v-if="videoinfo">
    <h1>{{ videoinfo.Name }}</h1>
    <div v-if="loading">Loading...</div>
    <span v-if="error" class="error">{{ error }}</span>
    <span v-if="croperror">{{ croperror }}</span>
    
    <div id="videoview-content" v-if="!loading" class="flex gap-2">
      <div id="column-1" class="w-[67vw]">
        <VideoPlayer class="relative" 
        v-if="!loading" v-bind:video-id="route.params.id" :video-src="videoPath" :mode="mode" :canvas-mode="canvasMode"
        :current-frame-nr="currentFrame" :videoinfo="videoinfo" :labeltype="labeltypes[selectedLabeltype]" :predicted-boxes="locationPredictions" :draw-predicted-boxes="selectedLocalizeModel"
        @play="updatePlaying" @pause="updatePaused" @seeked="onSeeked" @timeupdate="ontimeupdate"
        @add-box="onAddBox" @delete-box="onDeleteBox">
      </VideoPlayer>
      <SkillBalk v-if="modeIsSkills" :videoinfo="videoinfo" :Skills="skills" @skill-clicked="onSkillClicked" :currentFrame="currentFrame" class="mt-2"/>
      <SkillBalk v-if="modeIsSkills" :videoinfo="videoinfo" :Skills="predictions['skills']" @skill-clicked="onSkillClicked" :currentFrame="currentFrame" class="mt-2" :key="skillbalkKey"/>
      <div id="skill-controls" class="flex gap-2 my-2 wrap" v-show="modeIsSkills">
        <Button v-show="paused && !skillStore.selectedSkill.Id" @click="setFrameStart()">set frame Start</Button>
        <Button v-show="paused && !skillStore.selectedSkill.Id" @click="setFrameEnd()">set frame End</Button>
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
        <Button v-show="skillStore.selectedSkill.Id" @click="deselectSkill">Deselect skill</Button>
        <Button v-show="skillStore.selectedSkill.Id && skillStore.selectedSkill.FrameEnd != currentFrame" @click="frameToEndOfSkill">Frame to END of selected skill</Button>
        <Button v-show="skillStore.selectedSkill.Id && skillStore.selectedSkill.FrameEnd == currentFrame" @click="frameToStartOfSkill">Frame to START of selected skill</Button>
        <Button v-show="frameStart && frameEnd" @click="replaySection" v-shortkey="['r']" @shortkey="replaySection">Replay section (r)</Button>
        <Button v-show="skillStore.selectedSkill.Id" @click="playNextSection" v-shortkey="['n']" @shortkey="playNextSection" label="Play next section (n)" aria-label="Play next section (n)"></Button>
      </div>
      <div id="prediction-controls" class="flex gap-2 my-2 wrap" v-if="selectedSkillIsPrediction">
        <Button @click="splitPrediction">Split prediction</Button>
        <Button @click="mergeSplits">Merge</Button>
        <Button @click="() => shifPredictedSplitpoint(-1)">Shift splitpoint<i class="pi pi-arrow-left"></i></Button>
        <Button @click="() => shifPredictedSplitpoint(+1)">Shift splitpoint<i class="pi pi-arrow-right"></i></Button>
        <Button @click="acceptPredictedSkill">Accept<i class="pi pi-check"></i></Button>
      </div>
    </div>
    
    <div id="column-2" class="w-[33vw]">
      <div id="type-selection" class="flex h-fit gap-2 stretch">
          <Button :class="modeIsWatch ? 'p-button-highlight' : ''" @click="() => mode = 'WATCH'">Watch</Button>
          <Button :class="modeIsLocalize ? 'p-button-highlight' : ''" @click="() => mode = 'LOCALIZE'">Localize</Button>
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
          <div class="flex gap-2">
            <span class="my-auto">Labeltype</span>
            <Select v-model="selectedLabeltype" :options="Object.keys(labeltypes)"></Select>
            <InputNumber v-model="currentFrame" inputId="input-currentFrame" fluid></InputNumber>
          </div>
          <div class="mt-2 flex gap-2">
            <span class="my-auto">Canvas modus</span>
            <Select v-model="canvasMode" :options="canvasModes"></Select>
          </div>
          <div class="mt-2 flex gap-2">
            <span class="my-auto">Use</span>
            <Select v-model="selectedLocalizeModel" :options="Object.keys(localizeModelOptions)"></Select>
            <Select v-model="selectedWeights" :options="predictionWeightOptions"></Select>
            <Button v-if="selectedLocalizeModel && !localizeJobLaunched" @click="predictBoxes" label="Launch job"></Button>
            <span class="my-auto" v-if="localizeJobLaunched">Job in queue</span>
          </div>
        </div>
        
        <!--Skills -->
        <ConfirmPopup></ConfirmPopup>
        <SkillLabel v-if="modeIsSkills" :video-id="videoinfo.Id" :frame-start="frameStart" :frame-end="frameEnd"></SkillLabel>
        <div v-if="modeIsSkills" class="flex flex-wrap gap-2">
          <Button v-if="!skillStore.selectedSkillIsEmpty && skillStore.isNewSkill && frameStart && frameEnd" v-shortkey="['a']" @shortkey="addSkill" @click="addSkill" aria-label="Add skill (a)" label="Add skill (a)" icon="pi pi-plus-circle" class="my-2"></Button>
          <Button v-if="!skillStore.selectedSkillIsEmpty && !skillStore.isNewSkill && frameStart && frameEnd" v-shortkey="['u']" @shortkey="updateSkill" @click="updateSkill" aria-label="Update skill (u)" label="Update skill (u)" icon="pi pi-pencil" class="my-2"></Button>
          <Button v-if="!skillStore.selectedSkillIsEmpty && !skillStore.isNewSkill && frameStart && frameEnd" @click="confirmRemoveSkill($event)" severity="danger" aria-label="Delete skill" label="Delete skill" icon="pi pi-pencil" class="my-2"></Button>
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
import { getVideoInfo, getVideoPath, getCroppedVideoPath, removeVideoFrame, postVideoFrame, postSkill, putSkill, getSkillLevel, updateVideoSkillsCompleted, getVideoPredictions, getFrameLabelTypes, getJobOptions, launchJob, hasLocalizePredictions, getLocalizePredictions, deleteSkill } from '../services/videoService';
import { guidGenerator, sleep } from '@/helpers/utils';
import { onMounted, ref, watch, computed, toRaw } from 'vue'
import { useConfirm } from "primevue/useconfirm";
import { useRoute } from 'vue-router';
import { useSkillStore } from '@/stores/skillStore';
import LocalizeInfo from '@/components/LocalizeInfo.vue';
import SkillBalk from '@/components/SkillBalk.vue';
import SkillLabel from '@/components/SkillLabel.vue';
import VideoPlayer from '@/components/VideoPlayer.vue';

const confirm = useConfirm();
const route = useRoute()
const skillStore = useSkillStore()

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
const modeIsSkills = computed(() => mode.value == 'SKILLS')

const canvasModes = ['draw', 'edit', 'delete', 'accept']
const labeltypes = ref([])
const selectedLabeltype = ref(null)
const canvasMode = ref('draw')
const localizeModelOptions = ref(null)
const selectedLocalizeModel = ref(null)
const localizeJobLaunched = ref(false)
const locationPredictions = ref(null)

const currentFrame = ref(0)
const frameStart = ref(currentFrame.value)
const frameEnd = ref(undefined)

const predictions = ref({'boxes': [], 'skills': []})
const predictionWeightOptions = ['best', 'default']
const selectedWeights = ref('best')

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
const selectedSkillLevel = ref(0)
const selectedSkillIsPrediction = computed(() => skillStore.selectedSkill.hasOwnProperty('IsPrediction') && skillStore.selectedSkill.IsPrediction) 

watch(
  () => route.params.id,
  (newId) => (
    loadVideo(newId)
  )
)

const getPreviousPredictedSkill = (adjacent) => {
  if (adjacent) {
    return predictions.value['skills'].filter(s => s.FrameEnd == skillStore.selectedSkill.FrameStart)[0]
  } else {
    return predictions.value['skills'].filter(s => s.FrameEnd <= skillStore.selectedSkill.FrameStart).reduce((acc, current) => acc ? (current.FrameEnd > acc.FrameEnd && current.FrameEnd <= skillStore.selectedSkill.FrameStart ? current : acc) : current, null)
  }
}

const getNextPredictedSkill = (adjacent) => {
  if (adjacent) {
    return predictions.value['skills'].filter(s => s.FrameStart == skillStore.selectedSkill.FrameEnd)[0]
  } else {
    return predictions.value['skills'].filter(s => s.FrameStart >= skillStore.selectedSkill.FrameEnd).reduce((acc, current) => acc ? (current.FrameStart < acc.FrameStart && current.FrameStart >= skillStore.selectedSkill.FrameEnd ? current : acc) : current, null)
  }
}

const updateLevel = async () => {
  if (frameStart.value) {
    // let currentSkillinfo = normal2Reverse(skillStore.selectedSkill.ReversedSkillinfo)
    let previousSkillinfo = null
    let previousSkillname = null
    if (selectedSkillIsPrediction.value) {
      let previousSkill = getPreviousPredictedSkill(true)
      if (previousSkill) {
        previousSkillinfo = previousSkill['Skillinfo']
        previousSkillname = previousSkill['ReversedSkillinfo']['Skill']
      }
    }
    selectedSkillLevel.value = 0
    // TODO
    // await getSkillLevel(currentSkillinfo, previousSkillinfo, previousSkillname, frameStart.value, videoinfo.value.Id)
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

onMounted(async () => {
  getJobOptions('LOCALIZE').then(o => {
    localizeModelOptions.value = Object.fromEntries(Object.entries(o).filter(([model, details]) => details['base_model'] == 'YOLO'))
  })
  await loadVideo(videoId.value)
  await getFrameLabelTypes().then(types => {
    labeltypes.value = reverseDict(types)
    selectedLabeltype.value = types['1']
  })
  videoElement.value = document.getElementById("vid")
  getLocalizePredictions(videoinfo.value.Id).then(boxes => locationPredictions.value = boxes)
})

async function loadVideo(id) {
  loading.value = true;
  try {
    videoPath.value = await getVideoPath(id)
    videoinfo.value = await getVideoInfo(id)
    
    predictions.value = await getVideoPredictions(id).then(
      p => {
        console.log('skill predictions', p)
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
    .reduce((previous, current) => Math.min(previous, current.FrameNr), Infinity)
  let biggerFrameNr = videoinfo.value.Frames
    .filter((frameinfo) => frameinfo.FrameNr > currentFrame.value)
    .reduce((previous, current) => Math.min(previous, current.FrameNr), Infinity)
  currentFrame.value = biggerFrameNr == Infinity ? minFrameNr : biggerFrameNr

}

const setToPreviousFrame = () => {
  let maxFrameNr = videoinfo.value.Frames
    .reduce((previous, current) => Math.max(previous, current.FrameNr), -Infinity)
  let smallerFrameNr = videoinfo.value.Frames
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
  box['labeltype'] = Number(labeltypes.value[selectedLabeltype.value])
  await postVideoFrame(videoinfo.value.Id, Math.round(currentFrame.value), box).then(vi => videoinfo.value = vi).catch(e => error.value = e)
}

const onDeleteBox = async (box) => {
  await removeVideoFrame(videoinfo.value.Id, Math.round(currentFrame.value), box).then(vi => videoinfo.value = vi).catch(err => error.value = err)
}

const setFrameStart = () => { frameStart.value = skillStore.selectedSkill.FrameStart = currentFrame.value }
const setFrameEnd = () => { frameEnd.value = skillStore.selectedSkill.FrameEnd = currentFrame.value }

const play = () => {
  paused.value = false
  videoElement.value.play()
}

const onSkillClicked = (skillId, isPrediction) => {
  let skillsToFilter = isPrediction ? predictions.value['skills'] : skills.value

  let skill = skillsToFilter.filter(s => s.Id == skillId)[0]
  
  skillStore.setSelectedSkill(skill)
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
  setFrameEnd()
}

function deselectSkill() { 
  if (frameEnd.value) {
    frameStart.value = frameEnd.value
    frameEnd.value = undefined
  }
  if (!paused.value) {
    videoElement.value.pause()
  }
  currentFrame.value = frameStart.value
  skillStore.setSelectedSkill({ "FrameStart": frameStart.value, "Skillinfo": {} })
  videoElement.value.currentTime = frameStart.value / videoinfo.value.FPS
}

function frameToEndOfSkill() {
  currentFrame.value = skillStore.selectedSkill.FrameEnd
}

function frameToStartOfSkill() {
  currentFrame.value = skillStore.selectedSkill.FrameStart
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

function getNextSkill() {
  let skills = selectedSkillIsPrediction.value ? predictions.value['skills'] : videoinfo.value.Skills
  return skills
    .filter(skill => skill.FrameStart >= skillStore.selectedSkill.FrameEnd)
    .sort((a,b) => a.FrameEnd - b.FrameEnd)[0]
}

async function playNextSection() {
  let nextSkill = getNextSkill()
  if (nextSkill) {
    onSkillClicked(nextSkill.Id, selectedSkillIsPrediction.value)
    replaySection()
  }
}

function prepareNextLabel(fs) {
  let nextSkill = getNextSkill()
  if (nextSkill) {
    playNextSection()
    return
  } else {
    frameStart.value = fs
    frameEnd.value = undefined
    for (let skillIdx in skills.value) {
      skills.value[skillIdx].inCreation = false
    }
    deselectSkill()
  }
}

async function addSkill() {
  videoinfo.value = await postSkill(videoinfo.value.Id, skillStore.selectedSkill)
  prepareNextLabel(frameEnd.value)
}

async function updateSkill() {
  videoinfo.value = await putSkill(videoinfo.value.Id, skillStore.selectedSkill)
  prepareNextLabel(skillStore.selectedSkill.FrameEnd)
}

async function toggleSkillsCompleted() {
  updateVideoSkillsCompleted(videoinfo.value.Id, !videoinfo.value.Completed_Skill_Labels).then(() => videoinfo.value.Completed_Skill_Labels = ! videoinfo.value.Completed_Skill_Labels)
}

function shifPredictedSplitpoint(addFrames) {
  if (!selectedSkillIsPrediction.value) { return }
  if (currentFrame.value != skillStore.selectedSkill.FrameStart && currentFrame.value != skillStore.selectedSkill.FrameEnd) { return }
  
  let shiftedFrameNr = Math.round(currentFrame.value + addFrames)
  if (currentFrame.value == skillStore.selectedSkill.FrameEnd) {
    let nexSkill = getNextPredictedSkill(true)
    nexSkill.FrameStart = shiftedFrameNr
    skillStore.selectedSkill.FrameEnd = shiftedFrameNr
    frameEnd.value = shiftedFrameNr
  }

  if (currentFrame.value == skillStore.selectedSkill.FrameStart) {
    let previousSkill = getPreviousPredictedSkill(true)
    previousSkill.FrameEnd = shiftedFrameNr
    skillStore.selectedSkill.FrameStart = shiftedFrameNr
    frameStart.value = shiftedFrameNr
  }
  currentFrame.value = shiftedFrameNr
}

const acceptPredictedSkill = async () => {
  // skillStore.selectedSkill.Skillinfo = normal2Reverse(skillStore.selectedSkill.ReversedSkillinfo)
  // videoinfo.value = await postSkill(videoinfo.value.Id, {
  //   'FrameStart' : skillStore.selectedSkill.FrameStart,
  //   'FrameEnd' : skillStore.selectedSkill.FrameEnd,
  //   'Skillinfo' : skillStore.selectedSkill.Skillinfo
  // })
  // prepareNextLabel(frameEnd.value)
  // playNextSection()
  console.log('TODO : check')
}

const mergeSplits = () => {

  if (currentFrame.value == skillStore.selectedSkill.FrameEnd) {
    let nexSkill = getNextPredictedSkill(true)
    let nextFrameEnd = nexSkill.FrameEnd
    let idx = predictions.value["skills"].indexOf(nexSkill)
    predictions.value["skills"].splice(idx, 1)

    skillStore.selectedSkill.FrameEnd = nextFrameEnd
    frameEnd.value = nextFrameEnd
    currentFrame.value = nextFrameEnd
  }

  if (currentFrame.value == skillStore.selectedSkill.FrameStart) {
    let previousSkill = getPreviousPredictedSkill(true)
    let previsousFrameStart = previousSkill.FrameStart
    let idx = predictions.value["skills"].indexOf(previousSkill)
    predictions.value["skills"].splice(idx, 1)

    skillStore.selectedSkill.FrameStart = previsousFrameStart
    frameStart.value = previsousFrameStart
    currentFrame.value = previsousFrameStart
  }
}

const splitPrediction = async () => {
  let copy = structuredClone(toRaw(skillStore.selectedSkill))
  let splitpoint = Math.round((copy.FrameEnd + copy.FrameStart) / 2)
  
  skillStore.selectedSkill.FrameEnd = splitpoint
  frameEnd.value = splitpoint
  copy.FrameStart = splitpoint
  copy.Id = `${copy.Id}${splitpoint}`
  predictions.value["skills"].push(copy)
  currentFrame.value = splitpoint
  skillbalkKey.value += 1
}

const predictSkills = async () => {
  console.log("TODO") // TODO : launch job new router adaption
}

const predictBoxes = async () => {
  let jobarguments = {
    'type': 'PREDICT',
    'step': 'LOCALIZE',
    'videoId': videoinfo.value.Id,
    'weights': selectedWeights.value == 'default' ? localizeModelOptions.value[selectedLocalizeModel.value]['default_weights'] : 'best',
    'model': localizeModelOptions.value[selectedLocalizeModel.value]['model'],
    'model_kwargs' : localizeModelOptions.value[selectedLocalizeModel.value]
  }
  console.log('launching job', jobarguments)
  launchJob(jobarguments)
  localizeJobLaunched.value = true
  poll4Boxes()
}

const poll4Boxes = async () => {
  let noBoxes = true
  while (noBoxes) {
    hasLocalizePredictions(videoinfo.value.Id).then(hasBoxes => noBoxes = !hasBoxes)
    console.log('polling for boxes')
    await sleep(2000)
  }
  getLocalizePredictions(videoinfo.value.Id).then(boxes => console.log('boxes', locationPredictions.value = boxes))
  localizeJobLaunched.value = false
}

const confirmRemoveSkill = (event) => {
  confirm.require({
    target: event.currentTarget,
    message: 'Are you sure you want to proceed?',
    icon: 'pi pi-exclamation-triangle',
    rejectProps: {
      label: 'Cancel',
      severity: 'secondary',
      outlined: true
    },
    acceptProps: {
      label: 'Delete'
    },
    accept: () => {
      deleteSkill(videoId.value, frameStart.value, frameEnd.value)
      getVideoInfo(videoId.value).then(v => videoinfo.value = v)
      skillStore.setSelectedSkill({ "FrameStart": frameStart.value, "Skillinfo": {} })
    },
  });
};

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

