<template>
    <div class="flex flex-wrap gap-2 m-4">
        <span class="font-semibold my-auto">{{ name }}:</span>
        <Select v-if="isCategorical" 
            v-model="value" 
            :options="categoryOptions" 
            option-label="name" option-value="id"
        ></Select>
        <div v-if="isBoolean" class="flex flex-wrap gap-1">
            <RadioButton v-model="value" input-id="true" :value="true"></RadioButton>
            <label for="true">True</label>
            <RadioButton v-model="value" input-id="false" :value="false"></RadioButton>
            <label for="false">False</label>
        </div>
        <InputNumber v-if="isNumerical" class="w-20" inputClass="w-full" v-model="value" 
            :step="property['step']" 
            :min="property['min']" 
            :max="property['max']" 
            :minFractionDigits="0" 
            :maxFractionDigits="2" 
            show-buttons 
            :invalid="value % property['step'] !== 0" 
            :readonlyInput="true"
        ></InputNumber>
    </div>
</template>

<script setup>
import { useSkillStore } from '@/stores/skillStore';
import { computed, onMounted, ref, watch } from 'vue';

const skillStore = useSkillStore()
const props = defineProps({
    compositionName: {
        type: String,
        required: true
    },
    compositionIndex: {
        type: Number,
        required: true
    },
    name: {
        type: String,
        required: true,
    },
    stage: {
        type: String,
        required: true
    },
    stageNr: {
        type: String,
        required: false,
    },
    property: {
        type: Object,
        required: true
    }
})

const category = computed(() => props.property['type'])
const categoryOptions = computed(() => [ {id: 0, name:''}, ...props.property['categories']])
const isBoolean = computed(() => category.value == 'boolean')
const isCategorical = computed(() => category.value == 'categorical')
const isNumerical = computed(() => category.value == 'numerical')

const value = computed({
  get() {
    return props.stageNr
      ? skillStore.selectedSkill.Skillinfo[props.compositionName][props.compositionIndex][props.stage][props.stageNr][props.name]
      : skillStore.selectedSkill.Skillinfo[props.compositionName][props.compositionIndex][props.stage][props.name]
  },
  set(val) {
    if (props.stageNr) {
      skillStore.selectedSkill.Skillinfo[props.compositionName][props.compositionIndex][props.stage][props.stageNr][props.name] = val;
    } else {
      skillStore.selectedSkill.Skillinfo[props.compositionName][props.compositionIndex][props.stage][props.name] = val;
    }
  }
})

</script>

