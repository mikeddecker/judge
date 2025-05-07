<template>
  <div>
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
import { getFolder } from '../services/videoService';
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
  return {
    'localization': {
      'accuracy-mAP50' : 96.5,
      'accuracy-mAP50-95' : 66.6,
      'best-model' : 'YOLOv11',
      'train-images' : 732,
      'val-images' : 161,
      'test-images' : 130,
      'test-dd3-iou' : 0.77,
      'val-dd3-iou' : 0.81,
      'test-sr2-iou' : 0.12,
      'history' : [], // all runs?
      'train-time' : 1080.1
    },
    'segmentation' : [
      {
        'mse-val' : 0.06,
        'mse-test' : 0.065,
        'test-iou-real-over-predict' : 0.7,
        'test-iou-predict-over-real' : 0.71,
        'test-avg-splitpoint-distance' : 7.31,
        'test-avg-highest-splitpoint-distance' : 182,
        'test-highest-splitpoint-distance': 1502,
        'val-iou-real-over-predict' : 0.7,
        'val-iou-predict-over-real' : 0.71,
        'val-avg-splitpoint-distance' : 7.31,
        'val-avg-highest-splitpoint-distance' : 182,
        'val-highest-splitpoint-distance': 1502,
        'videos-train' : 44,
        'videos-val' : 7,
        'videos-test': 3,
        'total-frames': 7800,
        'best-model' : 'MViT',
        'train-time' : 6852.3
      }
    ],
    'recognition' : {
      'best-model' : 'MViT',
      'selected-model': 'MViT',
      'train-time' : 6852.3,
      'f1-scores-val' : {
        0: {
          "Type" : 0.5,
          "Rotations" : 0.5,
          "Turner1" : 0.5,
          "Turner2" : 0.5,
          "Skill" : 0.5,
          "Hands" : 0.5,
          "Feet" : 0.5,
          "Turntable" : 0.5,
          "BodyRotations" : 0.5,
          "Backwards" : 0.5,
          "Sloppy" : 0.5,
          "Hard2see" : 0.5,
          "Fault" : 0.5,
          "Total": 0.5,
        },
        1: {
          "Type" : 0.51,
          "Rotations" : 0.5,
          "Turner1" : 0.5,
          "Turner2" : 0.51,
          "Skill" : 0.55,
          "Hands" : 0.5,
          "Feet" : 0.5,
          "Turntable" : 0.5,
          "BodyRotations" : 0.5,
          "Backwards" : 0.5,
          "Sloppy" : 0.5,
          "Hard2see" : 0.5,
          "Fault" : 0.5,
          "Total": 0.505,
        }
      },
      'f1-scores-test' : {
        "Type" : 0.51,
        "Rotations" : 0.5,
        "Turner1" : 0.5,
        "Turner2" : 0.51,
        "Skill" : 0.55,
        "Hands" : 0.5,
        "Feet" : 0.5,
        "Turntable" : 0.5,
        "BodyRotations" : 0.5,
        "Backwards" : 0.5,
        "Sloppy" : 0.5,
        "Hard2see" : 0.5,
        "Fault" : 0.5,
        "Total": 0.505,
      },
      'modelcomparison' : {
        'MViT' : {
          'accuracy': 0.13,
          'acc-skills': 0.31,
          'last-trained' : '2025-04-28'
        },
        'MViT_extra_dense' : {
          'accuracy': 0.133,
          'acc-skills': 0.3112,
          'last-trained' : '2025-04-28'
        }
      },
      'distributions': {
        'skills' : {
          'push-up': {
            'train': 432,
            'test': 0,
            'val': 12
          },
          'frog': {
            'train': 325,
            'test': 0,
            'val': 13
          },
          'jump': {
            'train': 1586,
            'test': 0,
            'val': 132
          }
        }
      }
    }
  }
}

</script>

<style scoped>
.error {
  color: red;
}
</style>
