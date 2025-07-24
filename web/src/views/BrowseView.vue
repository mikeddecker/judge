<script setup>
import FolderContainer from '@/components/FolderContainer.vue';
import VideoInfoContainer from '@/components/VideoInfoContainer.vue';
import { discoverDrive, getFolder } from '@/services/videoService';
import { onMounted, ref } from 'vue';

const count = ref(0)
const children = ref([])
const folderId = ref(0)
const folderName = ref("Storage drive")
const parentId = ref(0)
const videos = ref([])
const totalLabels1 = ref(0)
const totalFrames = ref(0)
const testLabels1 = ref(0)
const testPercentage = ref(0)
const currentLabelType = ref(2)
const completed = ref(0)

const changeFolder = (newFolderId) => {
  getFolder(newFolderId)
  .then(response => {
    children.value = response.Children;
    folderName.value = response.Name;
    videos.value = Object.values(response.Videos).sort((a, b) => b.Id - a.Id);
    count.value = response.VideoCount;
    parentId.value = response.Parent ? response.Parent.Id : 0;
    totalLabels1.value = Object.values(response.Videos).reduce((prevValue, currentVideoInfo) => prevValue + currentVideoInfo.LabeledFrameCount, 0)
    totalFrames.value = Object.values(response.Videos).reduce((prevValue, currentVideoInfo) => prevValue + currentVideoInfo.FrameLength, 0)
    testLabels1.value = Object.values(response.Videos).reduce((prevValue, currentVideoInfo) => prevValue + (currentVideoInfo.Id % 10 == 5 ? currentVideoInfo.LabeledFrameCount : 0), 0)

    completed.value = Object.values(response.Videos).filter((v) => v.Completed_Skill_Labels).length
  })
  .catch(error => {
    console.error('Error fetching data:', error);
  });
}

onMounted(async () => {
  changeFolder(folderId.value)
})
</script>

<template>
  <h1>Navigate {{ folderName }}</h1>
  <p>Videos: {{ count }}</p>
  <FolderContainer @changeFolder="changeFolder" v-bind:folders="children" v-bind:parent-id="parentId"/>
  <VideoInfoContainer v-bind:videos="videos"/>
  <Button class="my-2" icon="pi pi-server" @click="discoverDrive" label="Discover drive" aria-label="Discover drive"></Button>
  <div>
    <a href="https://www.flaticon.com/free-icons/folder" title="folder icons">Folder icons created by DinosoftLabs - Flaticon</a><br>
    <a href="https://www.flaticon.com/free-icons/tick" title="tick icons">Tick icons created by Roundicons - Flaticon</a>
    <a href="https://www.flaticon.com/free-icons/objects" title="objects icons">Objects icons created by Smashicons - Flaticon</a>
    <a href="https://www.flaticon.com/free-icons/3d-model" title="3d-model icons">3d-model icons created by Mihimihi - Flaticon</a>
  </div>
</template>

<style>
@media (min-width: 1024px) {

}
</style>
