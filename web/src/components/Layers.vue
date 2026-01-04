<template>
  <DataTable :key="tableKey" v-model:expandedRows="expandedRows" :value="layers" dataKey="id"
  @rowExpand="onRowExpand" @rowCollapse="onRowCollapse" tableStyle="min-width: 60rem">
    <Column expander style="width: 5rem" />
    <Column field="name" header="Name">
      <template #editor="{ data, field }">
        <InputText v-model="data.name" @update:model-value="(newName) => (data.id, newName)"></InputText>
      </template>
    </Column>
    <Column field="type" header="Type"></Column>

    <template #expansion="slotProps">
      <p v-if="slotProps.data.type == 'numerical'">Min: {{ slotProps.data.min }}</p>
      <p v-if="slotProps.data.type == 'numerical'">Max: {{ slotProps.data.max }}</p>
      <p v-if="slotProps.data.type == 'numerical'">Step: {{ slotProps.data.step ? slotProps.data.step : '?' }}</p>
      <Listbox v-if="slotProps.data.type == 'categorical'" v-model="selectedCategoricalValue" :options="slotProps.data.categories" optionLabel="name" class="w-full md:w-56" listStyle="max-height:250px">
        <template #option="slotProps">
          <span>{{ slotProps.option.name }}</span>
        </template>
      </Listbox>
      <IftaLabel v-if="slotProps.data.type == 'categorical'" class="my-2">
        <InputText :id="`new_layer_value_${slotProps.data.layerId}`" v-model="new_layer_value" variant="filled"/>
        <label for="new_layer_value">Add value</label>
      </IftaLabel>
      <Button v-if="!new_layer_value_is_empty" class="" icon="pi pi-database" @click="() => createNewLayerValue(slotProps.data.id)" label="Add layer value" aria-label="Add layer value"></Button>
      <span v-if="slotProps.data.type == 'boolean'">True or False</span>
    </template>

    <template #footer>
      <Divider/>
      <h4>Add a new layer:</h4>
      <div class="flex flex-wrap gap-2 mb-2">
        <IftaLabel v-if="new_layer_value_is_empty">
          <InputText id="new_layer" v-model="new_layer" variant="filled" :disabled="!new_layer_value_is_empty" />
          <label for="new_layer">New layer</label>
        </IftaLabel>

        <Select v-if="new_layer_value_is_empty" v-model="selectedType" :options="layerTypes"></Select>

        <IftaLabel v-if="selectedTypeIsNumeric">
          <InputNumber id="new_layer_min" class="w-fit" v-model="new_layer_min" :min-fraction-digits="0" :max-fraction-digits="100"/>
          <label for="new_layer_min">Minimum</label>
        </IftaLabel>

        <IftaLabel v-if="selectedTypeIsNumeric" class="w-fit">
          <InputNumber id="new_layer_max" v-model="new_layer_max" :min-fraction-digits="0" :max-fraction-digits="100"/>
          <label for="new_layer_max">Maximum</label>
        </IftaLabel>

        <IftaLabel v-if="selectedTypeIsNumeric">
          <InputNumber id="new_layer_step" v-model="new_layer_step" :min-fraction-digits="0" :max-fraction-digits="100" :step="0.01"/>
          <label for="new_layer_step">Step</label>
        </IftaLabel>

        <Button v-if="!new_layer_is_empty && selectedType && (!selectedTypeIsNumeric || selectedTypeIsNumericAndFilledIn)" class="" icon="pi pi-database" @click="createNewLayer" label="Add layer" aria-label="Add layer"></Button>
      </div>
      Total {{ layers ? layers.length : 0 }} layers
    </template>
  </DataTable>
</template>

<script setup>
import { isNullOrWhiteSpace } from '@/helpers/utils';
import { addLayer, getLayerCompositions, getLayers, getLayerTypes, updateLayer } from '@/services/videoService';
import { computed, onMounted, ref } from 'vue';

const layers = ref([])
const layerTypes = ref([])
const selectedCategoricalValue = ref(null)
const selectedType = ref('')
const selectedTypeIsNumeric = computed(() => selectedType.value == 'numerical')
const selectedTypeIsNumericAndFilledIn = computed(() => selectedTypeIsNumeric.value && new_layer_min.value != null && new_layer_max.value != null)
const new_layer = ref('')
const new_layer_value = ref('')
const new_layer_is_empty = computed(() => isNullOrWhiteSpace(new_layer.value))
const new_layer_value_is_empty = computed(() => isNullOrWhiteSpace(new_layer_value.value))
const new_layer_min = ref(null)
const new_layer_max = ref(null)
const new_layer_step = ref(null)
const expandedRows = ref([])
const tableKey = ref(0)

const layerCompositions = ref(null)

onMounted(async () => {
  getLayerTypes().then(types => layerTypes.value = types)
  getLayerCompositions().then(compositions => layerCompositions.value = compositions)
  refreshLayers()
})

const refreshLayers = async () => {
  await getLayers().then(l => layers.value = l)
}

const createNewLayer = async () => {
  await addLayer(
    new_layer.value, null, selectedType.value, 
    new_layer_min.value, new_layer_max.value, new_layer_step.value
  )
  await refreshLayers()
  new_layer.value = null
}

const createNewLayerValue = async (layerId, categories) => {
  await addLayer(new_layer_value.value, layerId)
  await refreshLayers()
  new_layer.value = null
  new_layer_value.value = null
  tableKey.value = tableKey.value + 1
}

const onRowExpand = (args) => {
  console.log(args.data)
}

const onRowCollapse = (args) => {
  console.log(args)
}

</script>

