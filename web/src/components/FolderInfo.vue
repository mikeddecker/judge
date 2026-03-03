<script setup>
import { computed } from 'vue'

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

const trainTestCSS = computed(() => {
  return '' // Decide later
  if (!props.folderinfo) { return 'bg-cyan-50' }
  return props.folderinfo.IsTrain ? 'bg-green-50' : 'bg-teal-100'
} )

</script>

<template>
  <div 
    class="folderinfo w-[120px] m-2 p-1 border-1 border-solid border-zinc-200 rounded-xl"
    :class="trainTestCSS"
    @click="$emit('changeFolder', folderId)">
    <div class="container p-1">
      <img src="@/assets/folder.png" alt="folder image" />
    </div>
    <p class="m-1">{{ title }}</p>
  </div>
</template>

<style scoped>
.folderinfo:hover{
  border: 1px solid var(--color-border);
  box-shadow: 0.5px 0.5px 3px var(--color-heading);
}

h2 {
  margin-bottom: 0.4rem;
  color: var(--color-heading);
  word-wrap: break-word;
}

.container img {
  object-fit: contain;
  height: 100%;
  width: 100%;
  margin: -0.5rem 0 -1rem 0;
}

@media (min-width: 1024px) {
}
</style>

