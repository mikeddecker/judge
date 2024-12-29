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
      children: [],
      folderId: 0,
      folderName: "Storage drive",
      parentId: 0,
      videos: [],
    };
  },
  methods: {
    changeFolder(newFolderId) {
      getFolder(newFolderId)
      .then(response => {
        this.children = response.Children;
        this.folderName = response.Name;
        this.videos = response.Videos;
        this.parentId = response.Parent ? response.Parent.Id : 0;
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
    <FolderContainer @changeFolder="changeFolder" v-bind:folders="children" v-bind:parent-id="parentId"/>
    <VideoInfoContainer v-bind:videos="videos"/>
    <a href="https://www.flaticon.com/free-icons/folder" title="folder icons">Folder icons created by DinosoftLabs - Flaticon</a>
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
