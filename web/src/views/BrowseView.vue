<script>
import VideoInfoContainer from '@/components/VideoInfoContainer.vue';
import { getFolder } from '@/services/videoService';
export default {
  components: {
    VideoInfoContainer,
  },
  data() {
    return {
      children: [],
      folderId: 2,
      folderName: "Storage drive",
      videos: [],
    };
  },
  mounted() {
    getFolder(this.folderId)
      .then(response => {
        console.log('response', response)
        this.children = response.Children;
        this.folderName = response.Name;
        this.videos = response.Videos;
        console.log("folderData", this.folderData)
      })
      .catch(error => {
        console.error('Error fetching data:', error);
      });
  }
};
</script>

<template>
  <div class="browse">
    <h1>Navigate videos</h1>
    <VideoInfoContainer v-bind:videos="videos"/>
  </div>
  <pre>{{ JSON.stringify(videos, undefined, 2) }}</pre>
</template>

<style>
@media (min-width: 1024px) {
  .flexcontainer {
    display: flex
  }
}
</style>
