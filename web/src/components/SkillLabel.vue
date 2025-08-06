<template>
    <h3>Skilllabel</h3>
    <div id="skillinfo">
        <span>Start = {{ frameStart }}<br></span>
        <span>End = {{ frameEnd }}</span>
    </div>
    <h3>Compositions</h3>
    <div class="flex flex-wrap gap-2" v-if="layercomposition">
    <!-- <div class="flex flex-wrap gap-2" v-if="layercomposition && frameStart && frameEnd"> -->
        <Button v-for="compositionName in Object.keys(layercomposition)" 
        :aria-label="compositionName" :label="compositionName" icon="pi pi-plus"
        severity="secondary" @click="() => skillStore.addComposition(compositionName)" size="small"></Button>
    </div>
    <Tabs class="mt-4">
      <TabList>
        <Tab v-if="!skillStore.selectedSkillIsEmpty" v-for="(compositionLabelValues, compositionName) in skillStore.selectedSkill.Skillinfo" :key="`General-${compositionName}`" :value="`tab-${compositionName}`">{{ compositionName }}</Tab>
      </TabList>
      <TabPanels>
        <TabPanel v-if="layercomposition" v-for="(compositionLabelValues, compositionName) in skillStore.selectedSkill.Skillinfo" :key="`General-${compositionName}`" :value="`tab-${compositionName}`">
            <Tabs>
                <TabList>
                    <Tab v-for="(label, idx) in compositionLabelValues" :key="`item-${compositionName}-${idx}`" :value="`item-${compositionName}-${idx}`">{{ idx }}</Tab>
                </TabList>
                <TabPanels>
                    <TabPanel v-for="(label, idx) in compositionLabelValues" :key="`item-panel-${compositionName}-${idx}`" :value="`item-${compositionName}-${idx}`" class="flex flex-wrap">
                        <Card v-for="(upperStage, i) in ['GeneralProperties', 'StartProperties', 'EndProperties']" class="m-2">
                            <template #header>{{ upperStage }}</template>
                            <template #content>
                                <LayerPropertyValueSelector v-for="(layerValue, layerKey) in layercomposition[compositionName][upperStage]" :composition-name="compositionName" :composition-index="idx" :name="layerKey" :stage="upperStage" :property="layercomposition[compositionName][upperStage][layerKey]['property']"></LayerPropertyValueSelector>
                            </template>
                        </Card>
                        <Card>
                            <template #header>{{ 'StageProperties' }}</template>
                            <template #content>
                                <Card v-for="(stageLabel, stageNr) in label['StageProperties']" class="m-2">
                                    <template #header>{{ stageNr }}</template>
                                    <template #content>
                                        <LayerPropertyValueSelector v-for="(layerValue, layerKey) in layercomposition[compositionName]['StageProperties'][stageNr]" :composition-name="compositionName" :composition-index="idx" :name="layerKey" stage="StageProperties" :stage-nr="stageNr" :property="layercomposition[compositionName]['StageProperties'][stageNr][layerKey]['property']"></LayerPropertyValueSelector>
                                    </template>
                                </Card>
                            </template>
                        </Card>
                        
                    </TabPanel>
                </TabPanels>
            </Tabs>
        </TabPanel>
      </TabPanels>
    </Tabs>
</template>

<script setup>
import { getLayerCompositions } from '@/services/videoService';
import { computed, onMounted, ref, watch } from 'vue';
import LayerPropertyValueSelector from './LayerPropertyValueSelector.vue';
import { useSkillStore } from '@/stores/skillStore';

const skillStore = useSkillStore()
const props = defineProps({
    videoId: {
        type: Number,
        required: true,
    },
    frameStart: {
        type: Number,
        required: false,
    },
    frameEnd: {
        type: Number,
        required: false,
    },
})
// selectedSkill is the selected skill in the skill bar, not a new skill in edit.
// a skill in edit can be the selected skill, or it can be a new skill.

const layercomposition = ref(null)

onMounted(async () => {
    await getLayerCompositions().then(l => layercomposition.value = l)
})

</script>

