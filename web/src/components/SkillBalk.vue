<template>
    <div class="skillbalk">
      <div 
        v-for="s in props.Skills" 
        :key="s.Id" 
        :style="getSkillSectionStyle(s)" 
        class="skill-section"
        @click="handleClick(s.Id, s.hasOwnProperty('IsPrediction') ? s.hasOwnProperty('IsPrediction') : false)">
      </div>
      <div v-show="currentFrame" :style="getSkillSectionStyle(currentFrame)"></div>
    </div>
</template>
  
<script setup>
import { computed } from 'vue';

const props = defineProps(["videoinfo", "Skills", "currentFrame"])
const FrameLength = computed(() => props.videoinfo ? props.videoinfo.FrameLength : 1000)
const emit = defineEmits(["skill-clicked"])

function getColor(inCreation, isPrediction, skill) {
  if (inCreation) {
    return 'purple'
  }
  
  if (!isPrediction) {
    return 'var(--color-nav)'
  }

  if (skill.ReversedSkillinfo.Skill == 'jump') {
    return skill.Skillinfo.Rotations == 1 && skill.ReversedSkillinfo.Turner1 == 'normal' && skill.ReversedSkillinfo.Turner2 == 'normal' ? 'darkkhaki' : 'khaki'
  }

  if (!['frog', 'pushup', 'return from power'].includes(skill.ReversedSkillinfo.Skill)) {
    return skill.Skillinfo.Rotations == 1 && skill.ReversedSkillinfo.Turner1 == 'normal' && skill.ReversedSkillinfo.Turner2 == 'normal' ? 'mediumblue' : 'navy'
  }

  return skill.Skillinfo.Rotations == 1 && skill.ReversedSkillinfo.Turner1 == 'normal' && skill.ReversedSkillinfo.Turner2 == 'normal' ? 'var(--color-nav)' : 'mediumaquamarine'
}

function getSkillSectionStyle(skill) {
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
function handleClick(skillId, isPrediction) {
    emit('skill-clicked', skillId, isPrediction);
}
</script>
  
<style scoped>
  .skillbalk {
    height: 30px;
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
  