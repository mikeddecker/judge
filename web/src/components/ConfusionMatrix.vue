<template>
    <h3 class="text-xl font-bold mb-2 mt-4 w-fit">{{ name }}</h3>
    <table v-if="matrixColumns">
        <tbody>
            <tr>
                <th class="border p-2 bg-gray-100"></th>
                <th :colspan="matrixColumns.length" class="text-center bg-gray-200 font-bold">
                    Predicted
                </th>
            </tr>
            <tr>
                <th class="border p-2 bg-gray-100">Actual</th>
                <th v-for="col in matrixColumns.length" :key="col" class="border p-2">
                    {{ col - 1 }}
                </th>
            </tr>
            <tr v-for="label in matrixColumns">
                <td class="text-center p-2 bg-gray-200">{{ transformedMatrix[label]['actual'] }}</td>
                <td class="text-center p-2" :style="getCellStyle(label, predictionCount)" v-for="predictionCount in matrixColumns">{{ transformedMatrix[label][predictionCount] }}</td>
            </tr>
        </tbody>
    </table>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const props = defineProps({
    name: {
        type: String,
        required: true,
    },
    confusion: {
        type: Array[Array],
        required: true
    }
})
const transformedMatrix = ref(null)
const matrixColumns = ref(null)

// Transform 2D array into row objects
const transformMatrix = (matrix) => {
    return matrix.map((row, rowIndex) => {
        let obj = { actual: rowIndex }
        row.forEach((val, colIndex) => {
            obj[colIndex] = val
        })
        return obj
      })
}

const getColumns = (matrix) => {
    return Object.keys(matrix)
}

// Heatmap cell style
const getCellStyle = (label, prediction) => {
    const value = transformedMatrix.value[label][prediction]
    const total = sum(Object.values(transformedMatrix.value[label])) - transformedMatrix.value[label]['actual']
    const intensity = total > 0 ? value / total : 0
    const bgColor = `rgba(${(1-intensity) * 255}, 200, 90, ${0.15 + intensity * 0.85})`
    return {
        backgroundColor: bgColor,
        color: intensity > 0.5 ? 'white' : 'black',
        fontWeight: value > 0 ? 'bold' : 'normal'
    }
}

onMounted(() => {
    transformedMatrix.value = transformMatrix(props.confusion)
    matrixColumns.value = getColumns(props.confusion)
});

function sum(array) {
  return array.reduce((partialSum, a) => partialSum + a, 0);
}
</script>

