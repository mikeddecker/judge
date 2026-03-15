<script setup>
import { getVideoImagePath, updateVideoTrainingStatus } from '@/services/videoService';
import { computed, onMounted, ref } from 'vue';
import { useRouter } from 'vue-router';
import ProgressBar from './ProgressBar.vue';
import FlexContainer from './FlexContainer.vue';

const props = defineProps(['title', 'videoId', 'info'])
const router = useRouter()

const imageUrl = ref('');
const isUpdating = ref(false);

// Determine effective training status and inheritance state
const trainingState = computed(() => {
  // Extract the correct field from API response.
  // API returns either IsTrain (explicit value) or expects to read from Folder.IsTrain
  // Note: We use getter pattern to access possible null values
  const videoIsTrainValue = props.info.IsTrain;
  
  // Check if video has explicit is_train value set (not null/undefined)
  if (videoIsTrainValue !== undefined && videoIsTrainValue !== null) {
    return {
      isTraining: videoIsTrainValue === 1 || videoIsTrainValue === true,
      isInherited: false,
      value: videoIsTrainValue === 1 ? 1 : 0
    };
  }
  
  // If null/undefined, use folder's IsTrain (defaults to true/1 if not specified)
  const folderIsTrain = props.info.Folder?.IsTrain !== undefined 
    ? props.info.Folder.IsTrain 
    : true;
  
  return {
    isTraining: folderIsTrain === 1 || folderIsTrain === true,
    isInherited: true,
    value: null
  };
});

// Simplified computed for backward compatibility with template
const isTraining = computed(() => trainingState.value.isTraining);

const cssColorClass = computed(() => { return isTraining.value ? 'trainvideo' : 'testvideo' })

// percentageCompleted target 10% of frames labeled
const percentageCompleted = computed(() => props.info.BoxCount)

const toggleTrainingStatus = async (event) => {
  event.stopPropagation(); // Prevent route navigation when clicking toggle
  
  if (isUpdating.value) return;
  
  try {
    isUpdating.value = true;
    
    // Cycle through three states: train (1) → test (0) → inherit (null) → train (1)
    let nextState;
    const current = props.info.IsTrain;
    
    if (current === null || current === undefined) {
      // Currently inheriting from folder → switch to training
      nextState = 1;
    } else if (current === 1 || current === true) {
      // Currently training → switch to testing
      nextState = 0;
    } else {
      // Currently testing → switch to inherit from folder
      nextState = null;
    }
    
    await updateVideoTrainingStatus(props.videoId, nextState);
    
    // Update local state to reflect the change
    props.info.IsTrain = nextState;
  } catch (error) {
    console.error('Failed to update video training status:', error);
  } finally {
    isUpdating.value = false;
  }
};

onMounted(async () => {
  try {
    imageUrl.value = await getVideoImagePath(props.videoId);
  } catch (error) {
    console.error('Error fetching image:', error);
  }
});
</script>

<template>
  <div class="videoinfo" :class="cssColorClass" v-on:click="() => router.push(`/video/${videoId}`)">
    <div class="container">
      <img v-if="imageUrl" :src="imageUrl" alt="Video thumbnail" />
      <p v-else>Loading image...</p>
    </div>
    <div class="info">
      <span>{{ title }}</span>
    </div>
    <FlexContainer>
      <Tag v-for="tag in info['Tags']" :value="tag.Name" severity="Info"/>
    </FlexContainer>
    <div class="flex bottom-left">
      <span class="flex-end">video {{ videoId }}</span>
      <img v-if="info.Completed_Skill_Labels" class="percentageCompleted" src="@/assets/checked.png" alt="folder image" />
    </div>
    
    <!-- Train/Test Toggle Button -->
    <div class="m-1 flex justify-center">
      <button 
        @click="toggleTrainingStatus" 
        :disabled="isUpdating"
        :class="{ 
          'bg-green-500 border-green-600 hover:bg-green-600': !trainingState.isInherited && trainingState.isTraining, 
          'bg-blue-500 border-blue-600 hover:bg-blue-600': !trainingState.isInherited && !trainingState.isTraining,
          'bg-gray-500 border-gray-600 hover:bg-gray-600': trainingState.isInherited
        }"
        class="px-2 py-1 border-2 border-solid rounded text-white font-bold cursor-pointer transition-all hover:scale-105 hover:shadow-md disabled:opacity-60 disabled:cursor-not-allowed text-sm"
        :title="trainingState.isInherited ? 'Click to set as Training' : (trainingState.isTraining ? 'Click to set as Testing' : 'Click to inherit from Folder')"
      >
        {{ trainingState.isInherited ? '📁 Inherits' : (trainingState.isTraining ? '🎯 Train' : '🧪 Test') }}
      </button>
    </div>
    
    <ProgressBar :bgcolor="'#29ab87'" :percentage-completed="percentageCompleted" />
  </div>
</template>

<style scoped>
.videoinfo {
  margin: 0.7%;
  padding: 0.2rem;
  padding-bottom: 1.7rem;
  position: relative;
  width: 31%;
  border: 1px solid var(--color-border);
  border-radius: 0.55rem;
  box-shadow: 0.5px 0.5px 3px var(--color-heading);
  display: flex;
  flex-direction: column;
}

.info {
  margin: 0 0.2rem;
}

h2 {
  margin-bottom: 0.4rem;
  color: var(--color-heading);
  word-wrap: break-word;
}
.container {
  display: flex;
}

.container img {
  object-fit: contain;
  height: 100%;
  width: 100%;
}

img.percentageCompleted {
  width: 20%;
  align-self: flex-end;
  margin-top: auto;
  margin-bottom: 0.4rem;
  margin-right: 0.2rem;
  margin-left: auto;
}

.videoinfo:hover {
  background-color: khaki;
}

.flex {
  display: flex;
}

.flex-end {
  align-self: flex-end;
}

.bottom-left {
  margin: auto 0.2rem 0.2rem 0.2rem;
}

.testvideo {
  background-color:aqua;
}

@media (min-width: 1024px) {
  .videoinfo {
    width: 23%;
  }
}
</style>

