<template>
    <Card class="px-4 py-2">
        <template #title>{{ title }}</template>
        <template #content>
            <div v-for="(value, prop) in recipe">
                <p><span class="text-emerald-600">{{ prop }}</span>: {{ value }}</p>
            </div>
        </template>
        <template #footer>
            <Button class="float-right" label="train" :aria-label="`train ${title}`" @click="onButtonTrain"></Button>
        </template>
    </Card>
</template>

<script setup>
import { launchJob } from '@/services/videoService'

const props = defineProps({
  recipe: {
    type: Object,
    required: true,
  },
  title: {
    type: String,
    required: true,
  },
  step: {
    type: String,
    required: true,
  }
})

const onButtonTrain = () => {
  let jobarguments = {
    'type': 'TRAIN',
    'step': props.step,
    'recipe': props.title,
  }
  launchJob(jobarguments)
}

</script>

