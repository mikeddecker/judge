<template>
    <div class="container">
      <iframe id="iframe" width="900" height="640" :src="embeddedSource" 
        frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
        allowfullscreen></iframe>
    </div>
  </template>
  
  <script setup>
  import { computed } from 'vue';
  
  const props = defineProps(['title', 'videoId', 'videoSrc', 'info'])
  const ytid = computed(() => extractYTid(props.videoSrc))
  const embeddedSource = computed(() => `https://www.youtube.com/embed/${ytid.value}`)
  
  function extractYTid(str) {
    let parts = str.split("=")
    let embedding = parts.length == 2 ? parts[1] : parts[0]
    return embedding
  }
  </script>
  
  <style scoped>
  .container {
    position: relative;
    display: flex;
    justify-content: left;
    flex-wrap: wrap;
    max-width: 100%;
  }
  
  video {
    max-width: 100%;
    max-height: 70vh;
  }
  
  .overlay-canvas {
    position: absolute;
    border: 1px solid red;
    top: 0;
    left: 0;
    /* width: 100%; */
    /* height: 100%; */
  }
  
  @media (min-width: 1024px) {
    video {
      max-height: 85vh;
    }
  }
  </style>
  