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
        <Tab value="diff-score-comparison">Judges</Tab>
      </TabList>
      <TabPanels>
        <TabPanel value="recognition">
          <ResultsRecognitionView v-if="recognitionStats" :results="recognitionStats"></ResultsRecognitionView>
        </TabPanel>
        <TabPanel value="segmentation">
          <ResultsSegmentationView v-if="segmentationStats" :results="segmentationStats"></ResultsSegmentationView>
        </TabPanel>
        <TabPanel value="localization">
          <ResultsLocalizationView v-if="localizeStats" :results="localizeStats"></ResultsLocalizationView>
        </TabPanel>
        <TabPanel value="diff-score-comparison">
          <ResultsJudgeScores v-if="judgeStats" :results="judgeStats"></ResultsJudgeScores>
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

const localizeStats = ref(null)
const segmentationStats = ref(null)
const recognitionStats = ref(null)
const judgeStats = ref(null)

onMounted(async () => {
  loading.value = true;
  try {
    getStatistics();
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

  getStats('localize', bkVideoIds.value).then(r => localizeStats.value = r)
  getStats('segmentation', bkVideoIds.value).then(r => segmentationStats.value = r)
  getStats('recognition', bkVideoIds.value).then(r => {
    console.log(r);
    recognitionStats.value = r
  })
  getStats('judge', bkVideoIds.value).then(r => {
    console.log('judge results', r);
    judgeStats.value = r
  })
}

</script>

<style scoped>
.error {
  color: red;
}
</style>
