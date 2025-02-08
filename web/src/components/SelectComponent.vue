<template>
    <div>
        <label>{{ title }}</label>
        <select 
        :id="skilltype" 
        v-model="selectedValue" 
        @change="handleChange"
        >
        <option 
            v-for="(optionValue, optionId) in options" 
            :key="optionId" 
            :value="optionId"
        >
            {{ optionValue }}
        </option>
        </select>
    </div>
</template>
  
<script setup>
import { defineProps, defineEmits, ref } from 'vue';
  
const props = defineProps(["skilltype", "options", "title", "defaultValue"]);

const emit = defineEmits(['update:selected']);

const selectedValue = ref(props.defaultValue);

const handleChange = () => {
    console.log(selectedValue.value)
    emit('update:selected', props.skilltype, selectedValue.value, props.options[selectedValue.value]);
};
</script>

<style scoped>
div {
    border: 1px solid black;
    border-radius: 0.2rem;
    width: fit-content;
    padding: 0.2rem;
    margin: 0.15rem;
}
label {
    display: block;
    margin-bottom: 2px;
}

select {
    padding: 4px;
    border: 1px solid #ccc;
}
</style>
  