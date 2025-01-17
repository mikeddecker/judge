<script>
import FolderContainer from '@/components/FolderContainer.vue';
import VideoInfoContainer from '@/components/VideoInfoContainer.vue';
import { getFolder } from '@/services/videoService';
export default {
  components: {
    FolderContainer,
    VideoInfoContainer,
  },
  data() {
    return {
      count: 0,
      children: [],
      folderId: 0,
      folderName: "Storage drive",
      parentId: 0,
      videos: [],
      totalLabels: 0,
      totalFrames: 0,
      testLabels: 0,
      testPercentage: 0,
    };
  },
  methods: {
    changeFolder(newFolderId) {
      getFolder(newFolderId)
      .then(response => {
        this.children = response.Children;
        this.folderName = response.Name;
        this.videos = Object.values(response.Videos).sort((a, b) => a.FramesLabeledPerSecond - b.FramesLabeledPerSecond);
        this.count = response.VideoCount;
        this.parentId = response.Parent ? response.Parent.Id : 0;
        this.totalLabels = Object.values(response.Videos).reduce((prevValue, currentVideoInfo) => prevValue + currentVideoInfo.LabeledFrameCount, 0)
        this.totalFrames = Object.values(response.Videos).reduce((prevValue, currentVideoInfo) => prevValue + currentVideoInfo.FrameLength, 0)
        this.testLabels = Object.values(response.Videos).reduce((prevValue, currentVideoInfo) => prevValue + (currentVideoInfo.Id % 10 == 5 ? currentVideoInfo.LabeledFrameCount : 0), 0)
        this.testPercentage = Math.round(this.testLabels / this.totalLabels * 100)
      })
      .catch(error => {
        console.error('Error fetching data:', error);
      });
    }
  },
  mounted() {
    this.changeFolder(this.folderId)
  },
};
</script>

<template>
  <div class="browse">
    <h1>Navigate videos : {{ folderName }}</h1>
    <p>Videos: {{ count }}</p>
    <p>Labeled frames : {{ totalLabels }} / {{ totalFrames }} totalFrames</p>
    <p>Of which are test labels: {{ testLabels }} ({{ testPercentage }}%)</p>
    <p>First goal: label 1 frame of each second</p>
    <FolderContainer @changeFolder="changeFolder" v-bind:folders="children" v-bind:parent-id="parentId"/>
    <VideoInfoContainer v-bind:videos="videos"/>
    <a href="https://www.flaticon.com/free-icons/folder" title="folder icons">Folder icons created by DinosoftLabs - Flaticon</a>
  </div>
</template>

<style>
@media (min-width: 1024px) {
  .flexcontainer {
    display: flex
  }
}
</style>
