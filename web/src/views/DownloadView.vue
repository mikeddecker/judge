<template>
  <div>
    <h1>Download page</h1>
    <p>Provide a youtube url or youtube id</p>
    <input id="ytinput" v-model="ytURL" :placeholder="ytURL" />
    <div class="container">
      <div class="ytcomponent">
        <YTplayer :video-src="ytURL"></YTplayer>
      </div>
      
      <div class="info">
        <div class='buttongroup'>
          <button :class="coloredButtonClass(discipline, 'DD3')" @click="setDiscipline('DD3')">DD3</button>
          <button :class="coloredButtonClass(discipline, 'DD4')" @click="setDiscipline('DD4')">DD4</button>
          <button :class="coloredButtonClass(discipline, 'SR1')" @click="setDiscipline('SR1')">SR1</button>
          <button :class="coloredButtonClass(discipline, 'SR2')" @click="setDiscipline('SR2')">SR2</button>
          <button :class="coloredButtonClass(discipline, 'SR4')" @click="setDiscipline('SR4')">SR4</button>
          <button :class="coloredButtonClass(discipline, 'livestream')" @click="setDiscipline('livestream')">livestream</button>
        </div>
        <p>Year</p>
        <input id="yearinput" v-model="year"/>
        <div class='buttongroup'>
          <button :class="coloredButtonClass(year, 2020)" @click="setYear(2020)">2020</button>
          <button :class="coloredButtonClass(year, 2021)" @click="setYear(2021)">2021</button>
          <button :class="coloredButtonClass(year, 2022)" @click="setYear(2022)">2022</button>
          <button :class="coloredButtonClass(year, 2023)" @click="setYear(2023)">2023</button>
          <button :class="coloredButtonClass(year, 2024)" @click="setYear(2024)">2024</button>
          <button :class="coloredButtonClass(year, 2025)" @click="setYear(2025)">2025</button>
        </div>
        <p>Club</p>
        <input id="clubinput" v-model="club"/>
        <p>Name / info / annotation</p>
        <input id="infoinput" v-model="info" />
        <h2>{{ videoname }}</h2>
        <button :disabled="!downloadable" @click="downloadVideo">Download</button>
      </div>
    </div>
    
    <p>{{ downloadInfo }}</p>
  </div>
</template>

<script setup>
import YTplayer from '@/components/YTplayer.vue';
import { computed, ref } from 'vue';

const ytURL = ref('')
const year = ref(2025)
const discipline = ref('DD3')
const club = ref('sipiro')
const info = ref('')
const videoname = computed(() => [year.value, discipline.value, club.value, info.value.toLowerCase().replaceAll(' ','-')].join('-'))
const downloadable = computed(() => info.value != '' && ytURL.value != '')
const downloadInfo = ref('Here comes download info')

function coloredButtonClass(currentVal, buttonValue) { return currentVal == buttonValue ? 'btn-highlight' : 'btn-normal' }
function setDiscipline(d) { discipline.value = d }
function setYear(y) { year.value = y }
async function downloadVideo() {
  // TODO : post request?
  downloadInfo.value = `Started downloading ${videoname.value} from ${ytURL.value}`
  info.value = ''
  ytURL.value = ''
}
</script>

<style scoped>
h1 {
  margin: 0.5rem 0;
}
h2 {
  margin-top: 0.8rem;
}
.error {
  color: red;
}
input {
  width: 500px;
}
.container {
  margin-top: 0.8rem;
  display: flex;
}
.info {
  margin: 0.8rem;
}
.btn-highlight {
  background-color: green;
  color: aliceblue;
}
.btn-normal {
  background-color: darkseagreen;
}
button {
  font-size: 1.2rem;
  margin: 4px 2px;
  cursor: pointer;
}
input {
  font-size: 1rem;
  line-height: 1.6rem;
  padding: 0.25rem;
}
</style>
