<template>
    <div class="flex flex-wrap gap-2 my-2" :class="classes">
        <slot></slot>
    </div>
</template>

<script setup>
const props = defineProps({
    classes: {
        type: Object,
        required: false
    }
})
</script>