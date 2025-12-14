<template>
    <div class="skillbalk min-h-4">
      <div 
        v-for="s in props.Skills" 
        :key="s.Id" 
        :style="getSkillSectionStyle(s)" 
        class="skill-section"
        @click="handleClick(s.Id, s.hasOwnProperty('IsPrediction') ? s.hasOwnProperty('IsPrediction') : false)">
      </div>
      <div v-show="currentFrame" :style="getSkillSectionStyle(currentFrame)"></div>
      <div v-for="frame in props.labeledFrames" :key="frame" :style="getFrameLineStyle(frame)"></div>
    </div>
</template>
  
<script setup>
import { computed } from 'vue';

const props = defineProps({
  videoinfo: {
    required: true,
    type: Object,
  },
  Skills: {
    type: Object,
    required: true,
  },
  currentFrame: {
    required: true,
    type: Number,
  },
  labeledFrames: {
    type: Array,
    default: () => []
  }
})

const FrameLength = computed(() => props.videoinfo ? props.videoinfo.FrameLength : 1000)
const emit = defineEmits(["skill-clicked"])

function getColor(inCreation, isPrediction, skill) {
  if (inCreation) {
    return 'purple'
  }
  
  if (!isPrediction) {
    return 'var(--color-nav)'
  }

  if (skill.Skillinfo.Turntable > 0) {
    return 'orange'
  }

  if (skill.ReversedSkillinfo.Skill == 'jump') {
    if (skill.ReversedSkillinfo.Type == 'Chinese Wheel') {
      return 'mistyrose'
    }
    return skill.Skillinfo.Rotations == 1 && skill.ReversedSkillinfo.Turner1 == 'normal' && skill.ReversedSkillinfo.Turner2 == 'normal' ? 'darkkhaki' : 'khaki'
  }
  
  if (!['frog', 'pushup', 'return from power'].includes(skill.ReversedSkillinfo.Skill)) {
    return skill.Skillinfo.Rotations == 1 && skill.ReversedSkillinfo.Turner1 == 'normal' && skill.ReversedSkillinfo.Turner2 == 'normal' ? 'mediumblue' : 'navy'
  }
  
  return skill.Skillinfo.Rotations == 1 && skill.ReversedSkillinfo.Turner1 == 'normal' && skill.ReversedSkillinfo.Turner2 == 'normal' ? 'var(--color-nav)' : 'mediumaquamarine'
}

function getSkillSectionStyle(skill) {
  if (!skill) {
    console.log(skill)
  }
  // Current frame
  if (Number.isInteger(skill)) {
    // current position
    const relativeStart = skill / FrameLength.value;
    const left = relativeStart * 100;
    return {
      width: `2px`,
      left: `${left}%`,
      position: 'absolute',
      height: '100%',
      backgroundColor: 'red',
      cursor: 'pointer',
  }}

  // Skills
  const relativeStart = skill.FrameStart / FrameLength.value;
  const relativeEnd = skill.FrameEnd / FrameLength.value;
  let width = (relativeEnd - relativeStart) * 100 + 0.001;
  const left = relativeStart * 100;
  const inCreation = skill.inCreation ? true : false
  if (skill.FrameStart == skill.FrameEnd) {
    width=0.001
  }

  let isPrediction = skill.hasOwnProperty("IsPrediction") && skill.IsPrediction
  return {
      width: `${width}%`,
      left: `${left}%`,
      position: 'absolute', // To align the sections within the skillbalk
      height: '100%',
      backgroundColor: getColor(inCreation, isPrediction, skill),
      cursor: 'pointer'
  };
}

function getFrameLineStyle(frame) {
  const relativeStart = frame / FrameLength.value;
  const left = relativeStart * 100;
  return {
    width: `1px`,
    left: `${left}%`,
    position: 'absolute',
    height: '100%',
    backgroundColor: 'black',
    opacity: 0.7
  };
}
</script>
  
<style scoped>
  .skillbalk {
    position: relative;
    background-color: darkkhaki;
  }
  
  .skill-section {
    display: inline-block;
    text-align: center;
    color: blue;
    padding-top: 5px;
    border: 1px solid pink;
  }
</style>

