<template>
    <div class="skillbalk">
      <div 
        v-for="s in props.Skills" 
        :key="s.Id" 
        :style="getSkillSectionStyle(s)" 
        class="skill-section"
        @click="handleClick(s.Id)">
      </div>
      <div v-show="currentFrame" :style="getSkillSectionStyle(currentFrame)"></div>
    </div>
</template>
  
<script setup>
import { computed } from 'vue';

const props = defineProps(["videoinfo", "Skills", "currentFrame"])
const FrameLength = computed(() => props.videoinfo ? props.videoinfo.FrameLength : 1000)
const emit = defineEmits(["skill-clicked"])

function getSkillSectionStyle(skill) {
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
  const relativeStart = skill.FrameStart / FrameLength.value;
  const relativeEnd = skill.FrameEnd / FrameLength.value;
  let width = (relativeEnd - relativeStart) * 100 + 0.001;
  const left = relativeStart * 100;
  const inCreation = skill.inCreation ? true : false
  if (skill.FrameStart == skill.FrameEnd) {
    width=0.001
  }
  
  return {
      width: `${width}%`,
      left: `${left}%`,
      position: 'absolute', // To align the sections within the skillbalk
      height: '100%',
      backgroundColor: inCreation ? 'purple' : 'var(--color-nav)',
      cursor: 'pointer'
  };
}
function handleClick(skillId) {
    emit('skill-clicked', skillId);
}
</script>
  
<style scoped>
  .skillbalk {
    margin-top: 0.5rem;
    width: 95%;
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
  