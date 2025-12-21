<template>
    <h3 class="text-xl font-bold mb-2 mt-4 w-fit">{{ name }}</h3>
    <!-- Keep matrixColumns for indexing -->
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
            <tr>
                <th class="border p-2 bg-gray-100">Actual</th>
                <th v-for="predictedHeader of headers" :key="predictedHeader" class="border p-2">
                    {{ predictedHeader }}
                </th>
            </tr>
            <tr v-for="(actualHeader, actualIndex) in headers">
                <td class="text-center p-2 bg-gray-200">{{ actualHeader }}</td>
                <td 
                    class="text-center p-2" 
                    :style="getCellStyle(actualIndex, predictedIndex)" 
                    v-for="(predictedHeader, predictedIndex) in headers"
                >{{ transformedMatrix[actualIndex][predictedIndex] }}</td>
            </tr>
        </tbody>
    </table>
</template>

<script setup>
import chroma from 'chroma-js';
import { ref, onMounted } from 'vue'

const props = defineProps({
    name: {
        type: String,
        required: true,
    },
    values: {
        type: Array[Array],
        required: true
    },
    headers: {
        type: Array,
        required: true,
    }
})
const transformedMatrix = ref(null)
const matrixColumns = ref(null)
const diagonalScale = chroma.scale(['#ff6061', '#77dd77']).mode('lab'); // red to green, soft
const offDiagonalScale = chroma.scale(['#aec6cf', '#ff6961']).mode('lab'); // blueish to red, soft

// Transform 2D array into row objects
const transformMatrix = (matrix) => {
    return matrix.map((row, rowIndex) => {
        let obj = {}
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
const getCellStyle = (actualIndex, predictionIndex) => {
    const value = transformedMatrix.value[actualIndex][predictionIndex]
    const total = sum(Object.values(transformedMatrix.value[actualIndex]))
    const intensity = Math.pow(total > 0 ? value / total : 0, 0.5);

    // Calculate background color using Chroma.js
    const bgColor = actualIndex === predictionIndex 
        ? diagonalScale(intensity).alpha(0.6 + 0.4 * intensity).css() // slightly transparent
        : offDiagonalScale(intensity).alpha(0.25 + 0.7 * intensity).css(); 

    const color1 = actualIndex === predictionIndex ? 'white' : 'black';
    const color2 = actualIndex === predictionIndex ? 'black' : 'white';

    return {
        backgroundColor: bgColor,
        color: intensity > 0.25 ? color1 : color2,
        fontWeight: value > 0 ? 'bold' : 'normal'
    };
}

onMounted(() => {
    transformedMatrix.value = transformMatrix(props.values)
    matrixColumns.value = getColumns(props.values)
});

function sum(array) {
  return array.reduce((partialSum, a) => partialSum + a, 0);
}
</script>

