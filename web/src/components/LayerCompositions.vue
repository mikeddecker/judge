<template>
  <DataTable v-model:expandedRows="expandedRows" :value="layerCompositionsMapped" key="compositionName" data-key="name"
  @rowExpand="onRowExpand" @rowCollapse="onRowCollapse" tableStyle="min-width: 60rem">
    <Column expander style="width: 5rem"/>
    <Column field="compositionName" header="Name"/>

    <template #expansion="slotProps">
      <LayerComposition :composition-name="slotProps.data.compositionName" :composition="slotProps.data.composition" @composition-saved="refreshLayers"></LayerComposition>
    </template>

    <template #footer>
      <Divider/>
      <h4>Add a new composition:</h4>
      <div class="flex wrap gap-2 mb-2">
        <IftaLabel>
          <InputText id="new_composition" v-model="new_composition" variant="filled" />
          <label for="new_composition">New composition</label>
        </IftaLabel>
      </div>
      <LayerComposition v-if="!new_composition_is_empty" :composition-name="new_composition" @composition-saved="new_composition = null"></LayerComposition>
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
const tableRefreshKey = ref(0)
const selectedLayerComposition = ref(null)
const selectedCategoricalValue = ref(null)
const selectedType = ref('')
const selectedTypeIsNumeric = computed(() => selectedType.value == 'numerical')
const selectedTypeIsNumericAndFilledIn = computed(() => selectedTypeIsNumeric.value && new_composition_min.value != null && new_composition_max.value != null)
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

const createNewComposition = async () => {
  layerCompositions.value[new_composition.value] = {}
  new_composition.value = null
  tableRefreshKey.value += 1
  console.log(tableRefreshKey.value)
}

const onRowExpand = (args) => {
  console.log(args.data)
}

const onRowCollapse = (args) => {
  console.log(args)
}

const onLayerNameChanged = (id, newName) => {
  console.log('update', newName)
  // updateLayer(id, newName, null)
  // tags.value.filter(t => t.Id == id)[0].Name = newName
}

</script>

