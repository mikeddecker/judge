<script setup>
import { computed, ref } from 'vue'
import { updateFolderTrainingStatus } from '@/services/videoService';

const props = defineProps({
  title: {
    required: true,
    type: String
  },
  folderId: {
    required: false,
    type: String
  },
  folderinfo: {
    required: false,
    type: Object
  }
})

const isUpdating = ref(false);

// Determine effective training status for folder
const trainingState = computed(() => {
  if (!props.folderinfo) {
    return {
      isTraining: true,
      isInherited: false,
      value: 1
    };
  }
  
  const folderIsTrainValue = props.folderinfo.IsTrain;
  
  if (folderIsTrainValue !== undefined && folderIsTrainValue !== null) {
    return {
      isTraining: folderIsTrainValue === 1 || folderIsTrainValue === true,
      isInherited: false,
      value: folderIsTrainValue === 1 ? 1 : 0
    };
  }
  
  // Default to training if not specified
  return {
    isTraining: true,
    isInherited: false,
    value: 1
  };
});

const trainTestCSS = computed(() => {
  return '' // Decide later
  if (!props.folderinfo) { return 'bg-cyan-50' }
  return props.folderinfo.IsTrain ? 'bg-green-50' : 'bg-teal-100'
} )

const toggleTrainingStatus = async (event) => {
  event.stopPropagation(); // Prevent folder navigation when clicking toggle
  
  if (isUpdating.value || !props.folderId) return;
  
  try {
    isUpdating.value = true;
    
    // Cycle through two states for folders: train (1) → test (0) → train (1)
    let nextState;
    const current = props.folderinfo?.IsTrain;
    
    if (current === 1 || current === true) {
      // Currently training → switch to testing
      nextState = 0;
    } else {
      // Currently testing or undefined → switch to training
      nextState = 1;
    }
    
    await updateFolderTrainingStatus(props.folderId, nextState);
    
    // Update local state to reflect the change
    if (props.folderinfo) {
      props.folderinfo.IsTrain = nextState;
    }
  } catch (error) {
    console.error('Failed to update folder training status:', error);
  } finally {
    isUpdating.value = false;
  }
};
</script>

<template>
  <div 
    class="w-32 m-2 p-1 border border-solid border-zinc-200 rounded-xl hover:border-blue-400 hover:shadow-md transition-all cursor-pointer"
    @click="$emit('changeFolder', folderId)">
    <div class="p-1">
      <img src="@/assets/folder.png" alt="folder image" class="object-contain h-full w-full" />
    </div>
    <p class="m-1 text-sm truncate">{{ title }}</p>
    
    <!-- Train/Test Toggle Button for Folder -->
    <div class="flex justify-center mt-1">
      <button 
        @click="toggleTrainingStatus" 
        :disabled="isUpdating"
        :class="{ 
          'bg-green-500 border-green-600 hover:bg-green-600': trainingState.isTraining, 
          'bg-blue-500 border-blue-600 hover:bg-blue-600': !trainingState.isTraining
        }"
        class="px-2 py-1 border rounded text-base font-medium cursor-pointer transition-all hover:scale-110 disabled:opacity-60 disabled:cursor-not-allowed"
        :title="trainingState.isTraining ? 'Click to set as Test' : 'Click to set as Training'"
      >
        {{ trainingState.isTraining ? '🎯' : '🧪' }}
      </button>
    </div>
  </div>
</template>

<style scoped>
/* Tailwind CSS handles all styling - minimal overrides needed */
</style>

