<template>
  <DataTable v-model:expandedRows="expandedRows" :value="layerCompositionsMapped" key="compositionName" data-key="compositionName"
  @rowExpand="onRowExpand" @rowCollapse="onRowCollapse" tableStyle="min-width: 60rem">
    <Column expander style="width: 5rem"/>
    <Column field="compositionName" header="Name"/>

    <template #expansion="slotProps">
      <LayerComposition :composition-name="slotProps.data.compositionName" :composition="slotProps.data.composition" @composition-saved="refreshLayers"></LayerComposition>
    </template>

    <template #footer>
      <Divider/>
      <h4>Add a new composition:</h4>
      <div class="flex flex-wrap gap-2 mb-2">
        <IftaLabel>
          <InputText id="new_composition" v-model="new_composition" variant="filled" />
          <label for="new_composition">New composition</label>
        </IftaLabel>
      </div>
      <LayerComposition v-if="!new_composition_is_empty" :composition-name="new_composition" @composition-saved="new_composition = null" @moved:property="refreshLayers"></LayerComposition>
    </template>
  </DataTable>
</template>

<script setup>
import { isNullOrWhiteSpace } from '@/helpers/utils';
import { addLayerComposition, getLayerCompositions } from '@/services/videoService';
import LayerComposition from './LayerComposition.vue';
import { computed, onMounted, ref } from 'vue';

const layerCompositions = ref(null)
const layerCompositionsMapped = ref(null)
const selectedType = ref('')
const selectedTypeIsNumeric = computed(() => selectedType.value == 'numerical')
const new_composition = ref('')
const new_composition_is_empty = computed(() => isNullOrWhiteSpace(new_composition.value))
const expandedRows = ref([])

onMounted(async () => {
  refreshLayers()
})

const refreshLayers = async () => {
  await getLayerCompositions().then(compositions => {
    layerCompositions.value = compositions
    layerCompositionsMapped.value = Object.entries(compositions).map(([compositionName, composition]) => ({ compositionName, composition}))
  })
}

const onRowExpand = (args) => {
  console.log(args.data)
}

const onRowCollapse = (args) => {
  console.log(args)
}

</script>

