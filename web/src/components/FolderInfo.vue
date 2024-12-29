<script setup>
defineProps(['title', 'folderId'])

</script>

<template>
  <div class="folderinfo" @click="$emit('changeFolder', folderId)">
    <div class="container">
      <img src="@/assets/folder.png" alt="folder image" />
    </div>
    <div class="info">
      <p>{{ title }}</p>
    </div>
  </div>
</template>

<style scoped>
.folderinfo {
  margin: 0.7%;
  padding: 0.2rem;
  width: 120px;
}

.folderinfo:hover{
  border: 1px solid var(--color-border);
  border-radius: 0.55rem;
  box-shadow: 0.5px 0.5px 3px var(--color-heading);
}

.info {
  margin: 0 0.2rem;
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
