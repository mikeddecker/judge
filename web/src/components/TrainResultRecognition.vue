<template>
    <h3>Validation results</h3>
    <ButtonSelect
        :items="visibleMetrics"
        :selected="selectedMetric"
        @update:selectedItem="newMetric => selectedMetric = newMetric"
    ></ButtonSelect>
    <Chart type="line" :data="selectedMetricChartData" :options="selectedMetricChartOptions" />
    <ConfusionMatrix
        v-for="(matrix, layername) in confusionMatrices"
        :name="layername" :values="matrix"
        :headers="confusionHeaders[layername]"
    ></ConfusionMatrix>
</template>

<script setup>
import { getColor } from '@/helpers/utils';
import ConfusionMatrix from './ConfusionMatrix.vue';
import { computed, ref } from 'vue';
import ButtonSelect from './ButtonSelect.vue';

const props = defineProps({
    trainresult : {
        type: Object,
        required: true
    }
})

const visibleMetrics = ['acc', 'f1', 'precision', 'recall']

const metricDictKey = computed(() => props.trainresult.trainStart < '2026-02-01T19:00:00' ? 'metric_per_prop' : 'metric_per_layer')
const metricAvgDictKey = computed(() => props.trainresult.trainStart < '2026-02-01T19:00:00' ? 'metric_avg_of_props' : 'metric_avg_of_layers')
const validationResults = computed(() => props.trainresult.epochs[props.trainresult.bestEpoch])
const confusionMatrices = computed(() => validationResults.value.validationResults[metricDictKey.value].confusion)
const confusionHeaders = computed(() => validationResults.value.validationResults.confusion_heads)

// Line chart
const selectedMetric = ref('f1')
const selectedMetricChartOptions = computed(() => getMetricChartOptions(selectedMetric.value))
const selectedMetricChartData = computed(() => {
    let selectedMetricLayerValuesOverTime = Object.values(props.trainresult.epochs).map((epochResult) => epochResult.validationResults[metricDictKey.value][selectedMetric.value])
    let selectedMetricAvgValuesOverTime = Object.values(props.trainresult.epochs).map((epochResult) => epochResult.validationResults[metricAvgDictKey.value][selectedMetric.value])
    const layers = Object.keys(selectedMetricLayerValuesOverTime['1'])
    const epochs = Object.keys(props.trainresult.epochs)

    // Create one dataset per class
    const datasets = layers.map(layer => ({
        label: layer,
        data: Object.values(selectedMetricLayerValuesOverTime).map(metricLayerValueAtEpochX => metricLayerValueAtEpochX[layer]),
        borderColor: getColor(layer),
        fill: false,
        cubicInterpolationMode: 'monotone',
        tension: 0.4,
    }))

    datasets.push({
        label: 'Total',
        data: Object.values(selectedMetricAvgValuesOverTime).map(metricLayerAvgAtEpochX => metricLayerAvgAtEpochX),
        borderColor: getColor('Total'),
        fill: false,
        cubicInterpolationMode: 'monotone',
        tension: 0.4,
    })

    return {
        labels: epochs,
        datasets
    }
})

const getMetricChartOptions = (metric) => {
    return {
        responsive: true,
        plugins: {
            title: {
            display: true,
            text: `${metric}-scores-validation`
            },
        },
        interaction: {
            intersect: false,
        },
        scales: {
            x: {
                display: true,
                title: {
                    display: true,
                    text: 'epoch'
                }
            },
            y: {
                display: true,
                title: {
                    display: true,
                    text: `${metric}-score`
                },
                suggestedMin: 0,
                suggestedMax: 1,
                position: 'right'
            },
        }
    }
}

</script>

