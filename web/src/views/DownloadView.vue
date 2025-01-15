<template>
  <div>
    <h1>Download page</h1>
    <p>Provide a youtube url or youtube id</p>
    <input id="ytinput" v-model="ytURL" :placeholder="ytURL" />
    <div class="container">
      <div class="ytcomponent">
        <YTplayer :video-src="extractYTid(ytURL)"></YTplayer>
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
        <button :disabled="!downloadable" @click="downloadYTVideo">Download</button>
      </div>
    </div>
    <p>{{ downloadInfo }}</p>
    <h2 v-if="downloadError" class="error">{{ downloadMsg }}</h2>
  </div>
</template>

<script setup>
import YTplayer from '@/components/YTplayer.vue';
import { downloadVideo } from '@/services/videoService';
import { computed, ref } from 'vue';

const ytURL = ref('')
const year = ref(2024)
const discipline = ref('SR1')
const club = ref('sipiro')
const info = ref('')
const downloadError = ref(false)
const downloadMsg = ref('')
const videoname = computed(() => [year.value, discipline.value, club.value, info.value.toLowerCase().replaceAll(' ','-')].join('-'))
const downloadable = computed(() => info.value != '' && ytURL.value != '')
const downloadInfo = ref('Here comes download info')

function coloredButtonClass(currentVal, buttonValue) { return currentVal == buttonValue ? 'btn-highlight' : 'btn-normal' }
function setDiscipline(d) { 
  discipline.value = d
}
function getFolderId() {
  switch(discipline.value) { // Yes bad practices, but currently hardcoded
    case 'DD3':
      return 3
    case 'DD4':
      return 4
    case 'SR1':
      return 5
    case 'SR2':
      return 6
    case 'SR4':
      return 7
    case 'livestream':
      return 17
    default:
      return 0
  }
}
function setYear(y) { year.value = y }
function extractYTid(str) {
    let parts = str.split("=")
    let embedding = parts.length == 2 ? parts[1] : parts[0]
    return embedding
}
async function downloadYTVideo() {
  // TODO : post request?
  downloadInfo.value = `Started downloading ${videoname.value} from ${ytURL.value}`
  downloadError.value = false
  downloadVideo({
    'src' : 'yt',
    'URL' : extractYTid(ytURL.value),
    'name' : videoname.value,
    'folderId': getFolderId()
  }).catch(err => {
    downloadError.value = true
    downloadMsg.value = err
  })
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
.error {
  color: brown;
}
</style>
