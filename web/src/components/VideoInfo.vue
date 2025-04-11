<script setup>
import { getVideoImagePath } from '@/services/videoService';
import { computed, onMounted, ref } from 'vue';
import { useRouter } from 'vue-router';
import ProgressBar from './ProgressBar.vue';

const props = defineProps(['title', 'videoId', 'info'])
const router = useRouter()

const imageUrl = ref('');
const cssColorClass = computed(() => { return props.videoId % 10 == 5 ? 'testvideo' : 'trainvideo' })

// completed target 10% of frames labeled
// const labelthreshold = 0.1 // Minimun % to be labeled to reach 100%
// const completed = computed(() => Math.min(100, Math.floor(props.info.LabeledFrameCount / props.info.FrameLength / labelthreshold * 100)))
const completed = computed(() => props.info.LabeledFrameCount2)

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
    <div class="flex bottom-left">
      <span class="flex-end">video {{ videoId }}</span>
      <img v-if="info.Completed_Skill_Labels" class="completed" src="@/assets/checked.png" alt="folder image" />
    </div>
    <ProgressBar :bgcolor="'#29ab87'" :completed="completed" />
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

img.completed {
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
    width: 18%;
  }
}
</style>
