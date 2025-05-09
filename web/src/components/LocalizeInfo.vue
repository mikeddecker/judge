<script setup>
import { computed } from 'vue';

const props = defineProps(['videoinfo'])

const framesLabeledIndividual = computed(() => new Set(props.videoinfo.Frames.filter(box => box.LabelType == 2).map(box => box.FrameNr)).size)
const totalIndiviualBoxes = computed(() => props.videoinfo.LabeledFrameCount2)
const totalTeamBoxes = computed(() => props.videoinfo.LabeledFrameCount)

</script>

<template>
  <ul>
    <li>Team boxes: {{ totalTeamBoxes }}</li>
    <li>Individual boxes: {{ totalIndiviualBoxes }}</li>
    <li>Individual frames: {{ framesLabeledIndividual }}</li>
  </ul>
  </template>

<style scoped>


@media (min-width: 1024px) {
}
</style>
