<script>
import BrowseContainer from '@/components/BrowseContainer.vue';
export default {
  components: {
    BrowseContainer,
  },
};
</script>

<template>
  <div class="browse">
    <h1>This is the video browse page</h1>
    <BrowseContainer title="Bye bye"/>
    <BrowseContainer title="HELLOKES"/>
  </div>
</template>

<style>
@media (min-width: 1024px) {
  .browse {
    margin: 0 2rem;
    /* min-height: 100vh;
    display: flex;
    align-items: center; */
  }
  .flexcontainer {
    display: flex
  }
}
</style>
