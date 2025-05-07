<script setup>
const props = defineProps({
  results: {
    type: Object,
    required: true,
  },
})

const selected = props.results['selected-model']
const totalAccuracy = props.results["modelcomparison"][selected]['accuracy']
const skillAccuracy = props.results["modelcomparison"][selected]['acc-skills']

</script>


<template>
  <h2 class="mb-4">{{ selected }}</h2>
  <div class="flex gap-4">
    <Card>
      <template #header>Total Accuracy</template>
      <template #content>
        <span class="text-2xl -pt-12">{{ totalAccuracy }}%</span>
      </template>
    </Card>
    
    <Card>
      <template #header>Skill Accuracy</template>
      <template #content>
        <span class="text-2xl -pt-12">{{ skillAccuracy }}%</span>
      </template>
    </Card>
  </div>
  <pre>{{ results }}</pre>
</template>

<style scoped>
.p-card-body {
  padding-top: 0 !important;
}
</style>
