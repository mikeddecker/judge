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


const data = ref(null)
const loading = ref(true)
const error = ref('')
const videoId = ref(0)
const frameNr = ref(0)
const antwoord = ref('')
const recognitionResults = computed(() => data.value ? data.value['recognition'] : {})

onMounted(async () => {
  loading.value = true;
  try {
    data.value = await getStatistics();
  } catch {
    error.value = 'Failed To load';
  } finally {
    loading.value = false;
  }
})

async function getStatistics() {
  let stats =  await getStats('HAR_MViT')
  console.log(stats)
  return stats
}

</script>

<style scoped>
.error {
  color: red;
}
</style>
