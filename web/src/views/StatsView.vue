<template>
  <div class="w-full">
    <h1>Statistics</h1>
    <div v-if="error" class="error">{{ error }}</div>
    
    
    <div v-if="loading">Loading...</div>
    <Tabs v-else value="recognition">
      <TabList>
        <Tab value="recognition">Recognition</Tab>
        <Tab value="segmentation">Segmentation</Tab>
        <Tab value="localization">Localization</Tab>
      </TabList>
      <TabPanels>
        <TabPanel value="recognition">
          <ResultsRecognitionView :results="data['recognition']"></ResultsRecognitionView>
          <ResultsJudgeScores :results="data['scores']"></ResultsJudgeScores>
        </TabPanel>
        <TabPanel value="segmentation">
          <ResultsSegmentationView :results="data['segmentation']"></ResultsSegmentationView>
        </TabPanel>
        <TabPanel value="localization">
          <ResultsLocalizationView :results="data['localization']"></ResultsLocalizationView>
        </TabPanel>
      </TabPanels>
    </Tabs>

    

  </div>
</template>

<script setup>
import { getFolder, getStats } from '../services/videoService';
import { computed, onMounted, ref } from 'vue';
import ResultsSegmentationView from './ResultsSegmentationView.vue';
import ResultsRecognitionView from './ResultsRecognitionView.vue';
import ResultsLocalizationView from '@/views/ResultsLocalizationView.vue';
import ResultsJudgeScores from '@/views/ResultsJudgeScores.vue';

const data = ref(null)
const loading = ref(true)
const error = ref('')
const videoId = ref(0)
const frameNr = ref(0)
const antwoord = ref('')
const recognitionResults = computed(() => data.value ? data.value['recognition'] : {})
const bkVideoIds = ref([])

onMounted(async () => {
  loading.value = true;
  try {
    data.value = await getStatistics();
  } catch (e) {
    console.log(e)
    error.value = e;
  } finally {
    loading.value = false;
  }
})

async function getStatistics() {
  let videoIds = [2582, 2583]
  let maxId = 2590
  let minId = 2568
  bkVideoIds.value = [...Array(maxId - minId).keys()].map(i => i + minId)
  let stats =  await getStats('HAR_MViT', bkVideoIds.value)
  return stats
}

</script>

<style scoped>
.error {
  color: red;
}
</style>
