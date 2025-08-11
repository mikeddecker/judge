<template>
    <Chart ref="chartRef" type="bar" :data="values" :options="chartOptions" :class="class"/>
</template>

<script setup>
import { ref } from 'vue';

const props = defineProps({
    values: {
        type: Object,
        required: true,
    },
    direction: {
        type: String,
        required: true
    },
    title: {
        type: String,
        required: true,
    },
    class: {
        type: String,
        required: false
    },
    squared: {
        type: Boolean,
        required: false,
        default: false,
    }
})

const chartRef = ref(null)
const chartOptions = {
    indexAxis: props.direction,
    scales: {
        x: {
            ticks: {
                callback: function(value) {
                    if (props.direction == 'x') { return props.values['labels'][value] }
                    return props.squared ? Number(Math.pow(value, 2).toString()) : Number(value.toString())
                }
            }
        },
        y: {
            beginAtZero: true,
            ticks: {
                callback: function(value) {
                    if (props.direction == 'y') { return props.values['labels'][value] }
                    return props.squared ? Number(Math.pow(value, 2).toString()) : Number(value.toString())
                }
            }
        }
    },
    plugins: {
        title: {
            display: true,
            text: props.title,
            font: {
                size: 20
            }
        },
        legend: {
            position: 'top'
        },
        tooltip: {
            callbacks: {
            label: function(context) {
                const sqrtVal = props.direction == 'y' ? context.parsed.x : context.parsed.y
                const originalVal = Math.round(sqrtVal * sqrtVal)
                return `Count: ${originalVal}`
            }
        }
    },
  }
};

</script>

