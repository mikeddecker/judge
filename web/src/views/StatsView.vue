<template>
  <div class="w-full">
    <div v-if="error" class="error">{{ error }}</div>
    
    <div v-if="loading">Loading...</div>
    <Tabs v-else value="general">
      <TabList>
        <Tab value="general">General</Tab>
        <Tab value="monitoring">📊 Monitoring</Tab>
        <Tab value="recognition">Recognition</Tab>
        <Tab value="segmentation">Segmentation</Tab>
        <Tab value="localization">Localization</Tab>
        <Tab value="diff-score-comparison">Judges</Tab>
      </TabList>
      <TabPanels>
        <TabPanel value="general">
          <StatsGeneralView/>
        </TabPanel>
        <TabPanel value="monitoring">
          <StatsMonitoringView/>
        </TabPanel>
        <TabPanel value="localization">
          <StatsLocalizationView/>
          <ResultsLocalizationView/>
        </TabPanel>
        <TabPanel value="segmentation">
          <StatsSegmentationView/>
          <ResultsSegmentationView/>
        </TabPanel>
        <TabPanel value="recognition">
          <StatsRecognitionView/>
          <ResultsRecognitionView/>
        </TabPanel>
        <TabPanel value="diff-score-comparison">
          <ResultsJudgeView/>
        </TabPanel>
      </TabPanels>
    </Tabs>

  </div>
</template>

<script setup>
import { getFolder, getFrameLabelTypes, getResults, getStats } from '../services/videoService';
import { computed, onMounted, ref } from 'vue';
import StatsGeneralView from './StatsGeneralView.vue';
import StatsMonitoringView from './StatsMonitoringView.vue';
import StatsLocalizationView from './StatsLocalizationView.vue';
import StatsRecognitionView from './StatsRecognitionView.vue';
import StatsSegmentationView from './StatsSegmentationView.vue';
import ResultsLocalizationView from './ResultsLocalizationView.vue';
import ResultsJudgeView from './ResultsJudgeView.vue';
import ResultsRecognitionView from './ResultsRecognitionView.vue';
import ResultsSegmentationView from './ResultsSegmentationView.vue';

const loading = ref(true)
const error = ref('')

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

// Move fetch to the dedicated view.
async function getStatistics() {
}

</script>

<style scoped>
.error {
  color: red;
}
</style>

