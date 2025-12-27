<template>
    <Card class="px-4 py-2">
        <template #title>{{ title }}</template>
        <template #content>
            <div v-for="(value, prop) in recipe">
                <p><span class="text-emerald-600">{{ prop }}</span>: {{ value }}</p>
            </div>
            <div class="flex flex-wrap gap-4">
              <div class="flex items-center gap-2">
                  <RadioButton v-model="testrun" inputId="true" name="True" :value="true" />
                  <label for="true">test</label>
              </div>
              <div class="flex items-center gap-2">
                  <RadioButton v-model="testrun" inputId="false" name="False" :value="false" />
                  <label for="false">run</label>
              </div>
          </div>
        </template>
        <template #footer>
            <Button class="float-right" label="train" :aria-label="`train ${title}`" @click="onButtonTrain"></Button>
        </template>
    </Card>
</template>

<script setup>
import { launchJob } from '@/services/videoService'
import { ref } from 'vue'

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

const testrun = ref(false)

const onButtonTrain = () => {
  let jobarguments = {
    'type': 'TRAIN',
    'step': props.step,
    'recipe': props.title,
    'testrun': testrun.value,
  }
  launchJob(jobarguments)
}

</script>

