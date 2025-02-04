<script setup>
import { computed } from 'vue';

const props = defineProps(['frameinfo'])
const emit = defineEmits(["deleteBox"])

const color = computed(() => props.frameinfo.color)

</script>

<template>
    <button :class="color" @click="emit('deleteBox', frameinfo)"><img src="@/assets/delete.png" alt="buttonpng" class="icon"/></button>
    <!-- <p>{{ frameinfo.LabelType }}</p> -->
</template>

<style scoped>
button {
  min-width: 5rem;
  height: 3rem;
  margin: 0.1rem;
  border-radius: 10%;
}
.blue { background-color: blue; }
.white { background-color: white; }
.pink { background-color: pink; }
.yellow { background-color: yellow; }

img {
  max-width: 50%;
  max-height: 50%;
}

@media (min-width: 1024px) {
}
</style>
