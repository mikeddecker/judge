<script setup>
import { computed, ref } from 'vue';
import BoxCard from './BoxCard.vue';
import BoxContainer from './BoxContainer.vue';

const props = defineProps(['videoinfo', 'currentFrame', 'buttonColors'])

const localizeModes = ['review', 'draw', 'edit']
const localizeMode = ref('review')
const localizeModeTeamOrIndividual = ref('individual')
const modeIsReview = computed(() => localizeMode.value == 'review')
const modeIsDraw = computed(() => localizeMode.value == 'draw')
const modeIsEdit = computed(() => localizeMode.value == 'edit')
const currentBoxes = computed(() => props.videoinfo.Frames)

const deleteLabel = (box) => {
  console.log(box)
}

</script>

<template>
  <div class="w-full flex gap-2">
    <Button v-if="modeIsReview"><</Button>
    <Button v-if="modeIsReview">></Button>
  </div>
  <div class="w-full mt-1">
    <BoxCard class="icon" v-for="(box, index) in currentBoxes" :key="index" :frameinfo="box" :bg-color="buttonColors[index]" @deleteBox="deleteLabel"/>
  </div>
  {{ currentBoxes }}
</template>

<style scoped>
.p-button, .arrow {
  font-size: x-large;
}

.icon {
  padding: 0;
}

@media (min-width: 1024px) {
}
</style>
