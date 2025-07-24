<script setup>
import { computed } from 'vue';

const props = defineProps(['videoinfo'])

const numForegroundBoxes = computed(() => props.videoinfo.Frames.filter(box => box.LabelType == 1).length)
const numBackgroundBoxes = computed(() => props.videoinfo.Frames.filter(box => box.LabelType == 2).length)
const numFramesLabeled = computed(() => props.videoinfo.LabeledFrameCount)

</script>

<template>
  <ul>
    <li>Frames with boxes: {{ numFramesLabeled }}</li>
    <li>Total boxes: {{ props.videoinfo.Frames.length }}</li>
    <li>Foreground boxes: {{ numForegroundBoxes }}</li>
    <li>Background boxes: {{ numBackgroundBoxes }}</li>
  </ul>
</template>

<style scoped>


@media (min-width: 1024px) {
}
</style>
