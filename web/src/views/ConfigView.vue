<template>
  <h1>Config</h1>
  
  <h2 class="mt-6 mb-2">Tags</h2>
  <TagConfig/>

  <h2 class="mt-6 mb-2">Localization</h2>
  <Listbox :options="frameLabelTypes" disabled class="w-full md:w-56"></Listbox>
  
  <h2 class="mt-6 mb-2">Recognition</h2>
  <LayerConfig/>
</template>

<script setup>
import LayerConfig from '@/components/LayerConfig.vue';
import TagConfig from '@/components/TagConfig.vue';
import { getFrameLabelTypes } from '@/services/videoService';
import { computed, onMounted, ref } from 'vue';

const frameLabelTypes = ref([])

onMounted(async () => {
  getFrameLabelTypes().then(types => frameLabelTypes.value = Object.values(types))
})

</script>

<style>

</style>

