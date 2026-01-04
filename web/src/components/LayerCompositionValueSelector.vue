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
        :step="layer['step']" 
        :min="layer['min']" 
        :max="layer['max']" 
        :minFractionDigits="0" 
        :maxFractionDigits="2" 
        show-buttons 
        :invalid="innervalue == null" 
        :readonlyInput="true"
        ></InputNumber>
        <span class="my-auto">
            (
                Focus?
                <Checkbox v-model="innerfocussed" binary />
            )
        </span>
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
    layer: {
        type: Object,
        required: true
    },
    value: {
        required: true,
    },
    focussed: {
        type: Boolean,
        required: false
    }
})

const emit = defineEmits(['update:value', 'update:focussed'])

const category = computed(() => props.layer['type'])
const categoryOptions = computed(() => [ {id: 0, name:''}, ...props.layer['categories']])
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

const innerfocussed = computed({
  get() {
    return props.focussed
  },
  set(val) {
    emit('update:focussed', val)
  }
})

</script>

