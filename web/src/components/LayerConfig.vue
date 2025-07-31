<template>
  <h3>Layers</h3>
  <span>Keep in mind that the reactivity always lags one behind upon adding values.</span>
  <Layers/>

  <Divider/>
  <h3>LayerCompositions</h3>
  <LayerCompositions/>
</template>

<script setup>
import Layers from './Layers.vue';
import LayerCompositions from './LayerCompositions.vue';
</script>

