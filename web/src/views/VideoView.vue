<template>
  <div v-if="videoinfo">
    <h1>{{ videoinfo.Name }}</h1>
    <div v-if="loading">Loading...</div>
    <span v-if="error" class="error">{{ error }}</span>
    <span v-if="croperror">{{ croperror }}</span>
    
    <div id="videoview-content" v-if="!loading" class="flex gap-2">
      <div id="column-1" class="w-[75vw]">

        <VideoPlayer class=" relative" 
        v-if="!loading" v-bind:video-id="route.params.id" :video-src="videoPath" :mode="mode"
        :current-frame-nr="currentFrame" :videoinfo="videoinfo"
        @play="updatePlaying" @pause="updatePaused" @seeked="onSeeked" @timeupdate="ontimeupdate">
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
        <div>currentFrame: {{ Math.round(currentFrame) }}</div>
        <LocalizeInfo v-if="modeIsLocalize" :videoinfo="videoinfo"></LocalizeInfo>
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
import { getVideoInfo, getVideoPath, getCroppedVideoPath } from '../services/videoService';
import { onMounted, ref, watch, computed } from 'vue'
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

const mode = ref('WATCH')
const modeIsWatch = computed(() => mode.value == 'WATCH')
const modeIsLocalize = computed(() => mode.value == 'LOCALIZE')
const modeIsSegment = computed(() => mode.value == 'SEGMENT')
const modeIsSkills = computed(() => mode.value == 'SKILLS')

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


const sleep = ms => new Promise(r => setTimeout(r, ms));
watch(
  () => route.params.id,
  (newId) => (
    loadVideo(newId)
  )
)

onMounted(async () => {
  await loadVideo(videoId.value)
})
async function loadVideo(id) {
  loading.value = true;
  try {
    videoPath.value = await getVideoPath(id)
    videoinfo.value = await getVideoInfo(id)
  } catch {
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
  console.log("upontimeupdate", seconds)
  currentFrame.value = videoinfo.value.FPS * seconds
}

const onSkillClicked = (skillId) => {
  console.log('onSkillClicked', skillId)
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
