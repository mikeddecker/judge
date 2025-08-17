<template>
    <div class="flex flex-wrap gap-2 m-4">
        <span class="font-semibold my-auto">{{ name }}:</span>
        <Select 
            v-if="isCategorical" 
            :id="guidvalue"
            v-model="innervalue" 
            :options="categoryOptions" 
            option-label="name" option-value="id"
            :invalid="innervalue == null"
        ></Select>
        <div v-if="isBoolean" class="flex flex-wrap gap-1">
            <RadioButton v-model="innervalue" :input-id="`${guidvalue}-true`" :value="true" :invalid="innervalue == null"></RadioButton>
            <label :for="`${guidvalue}-true`">True</label>
            <RadioButton v-model="innervalue" :input-id="`${guidvalue}-false`" :value="false" :invalid="innervalue == null"></RadioButton>
            <label :for="`${guidvalue}-false`">False</label>
        </div>
        <InputNumber v-if="isNumerical" class="w-20" inputClass="w-full" v-model="innervalue" 
            :id="guidvalue"
            :step="property['step']" 
            :min="property['min']" 
            :max="property['max']" 
            :minFractionDigits="0" 
            :maxFractionDigits="2" 
            show-buttons 
            :invalid="innervalue == null" 
            :readonlyInput="true"
        ></InputNumber>
    </div>
</template>

<script setup>
import { guidGenerator } from '@/helpers/utils';
import { useSkillStore } from '@/stores/skillStore';
import { computed, onMounted, ref, watch } from 'vue';

const skillStore = useSkillStore()
const props = defineProps({
    name: {
        type: String,
        required: true,
    },
    property: {
        type: Object,
        required: true
    },
    value: {
        required: true,
    }
})

const emit = defineEmits(['update:value', 'update:focussed'])

const category = computed(() => props.property['type'])
const categoryOptions = computed(() => [ {id: 0, name:''}, ...props.property['categories']])
const isBoolean = computed(() => category.value == 'boolean')
const isCategorical = computed(() => category.value == 'categorical')
const isNumerical = computed(() => category.value == 'numerical')

const guidvalue = guidGenerator()

const innervalue = computed({
  get() {
    if (props.value == null || props.value == undefined) { return props.value }
    return props.value
  },
  set(val) {
    emit('update:value', val)
  }
})

</script>

