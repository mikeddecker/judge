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
      totalLabels1: 0,
      totalLabels2: 0,
      totalFrames: 0,
      testLabels1: 0,
      testLabels2: 0,
      testPercentage: 0,
      currentLabelType: 2,
      completed: 0,
    };
  },
  methods: {
    changeFolder(newFolderId) {
      getFolder(newFolderId)
      .then(response => {
        this.children = response.Children;
        this.folderName = response.Name;
        this.videos = Object.values(response.Videos).sort((a, b) => b.Id - a.Id);
        // this.videos = Object.values(response.Videos).sort((a, b) => a.FramesLabeledPerSecond - b.FramesLabeledPerSecond);        
        this.count = response.VideoCount;
        this.parentId = response.Parent ? response.Parent.Id : 0;
        this.totalLabels1 = Object.values(response.Videos).reduce((prevValue, currentVideoInfo) => prevValue + currentVideoInfo.LabeledFrameCount, 0)
        this.totalLabels2 = Object.values(response.Videos).reduce((prevValue, currentVideoInfo) => prevValue + currentVideoInfo.LabeledFrameCount2, 0)
        this.totalFrames = Object.values(response.Videos).reduce((prevValue, currentVideoInfo) => prevValue + currentVideoInfo.FrameLength, 0)
        this.testLabels1 = Object.values(response.Videos).reduce((prevValue, currentVideoInfo) => prevValue + (currentVideoInfo.Id % 10 == 5 ? currentVideoInfo.LabeledFrameCount : 0), 0)
        this.testLabels2 = Object.values(response.Videos).reduce((prevValue, currentVideoInfo) => prevValue + (currentVideoInfo.Id % 10 == 5 ? currentVideoInfo.LabeledFrameCount2 : 0), 0)
        this.testPercentage = Math.round(this.testLabels / this.totalLabels * 100)

        this.completed = Object.values(response.Videos).filter((v) => v.Completed_Skill_Labels).length
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
    <p>Completly labeled skills: {{ completed }}</p>
    <p>Total frames : {{ totalFrames }}</p>
    <p>Full team labels: {{ totalLabels1 }}</p>
    <p>Individual team labels: {{ totalLabels2 }}</p>
    <FolderContainer @changeFolder="changeFolder" v-bind:folders="children" v-bind:parent-id="parentId"/>
    <VideoInfoContainer v-bind:videos="videos"/>
    <a href="https://www.flaticon.com/free-icons/folder" title="folder icons">Folder icons created by DinosoftLabs - Flaticon</a>
    <a href="https://www.flaticon.com/free-icons/tick" title="tick icons">Tick icons created by Roundicons - Flaticon</a>
  </div>
</template>

<style>
@media (min-width: 1024px) {
  .flexcontainer {
    display: flex
  }
}
</style>
