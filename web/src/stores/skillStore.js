// Create a new store instance.
import { getLayerCompositions } from "@/services/videoService";
import { defineStore } from "pinia";
import { toRaw } from "vue";

export const useSkillStore = defineStore("skill", {
  state: () => {
    return {
      selectedSkill: { 'Skillinfo' : {} },
      layercomposition: undefined,
    };
  },
  actions: {
    setSelectedSkill(skill) { this.selectedSkill = skill },
    async addComposition(selectedCompositionName) {
      const getDefaultValue = (layerproperty) => {
        return Object.fromEntries(Object.entries(layerproperty).map(([propertyname, propertyinfo]) => {
          let dv = null
          // TODO create default options in config
          switch (propertyinfo.property.type) {
            case 'numerical': dv = propertyinfo.property.min; break;
            case 'categorical': dv = 0; break;
            case 'boolean': dv = false; break;
          }
          return [propertyname, dv]
        }))
      }
      let selectedLayerComposition = this.layercomposition[selectedCompositionName]
      let label = Object.fromEntries(Object.entries(selectedLayerComposition).map(([upperStage, upperStageValues]) => {
        if (upperStage == 'compositionName') { return [ upperStage, upperStageValues ]}
        if (upperStage == 'StageProperties') {
          return [upperStage, Object.fromEntries(Object.entries(upperStageValues).map(([stageNr, stageValues]) => {
              return [stageNr, getDefaultValue(stageValues)]
          }))]
        } else {
          return [upperStage, getDefaultValue(upperStageValues)]
        }
      }));
      
      (this.selectedSkill['Skillinfo'][selectedCompositionName] ??= []).push(label); // Create or push to the array
    },
    async loadData() {
      getLayerCompositions().then(l => this.layercomposition = l)
    },
    duplicateCompositionValues(compositionName, index) {
      const valuesToCopy = toRaw(this.selectedSkill.Skillinfo[compositionName][index])
      for (let i = 0; i < this.selectedSkill.Skillinfo[compositionName].length; i++) {
        this.selectedSkill.Skillinfo[compositionName][i] = structuredClone(valuesToCopy)
      }
    },
    deleteCompositionValues(compositionName, index) {
      delete this.selectedSkill.Skillinfo[compositionName].splice(index, 1)
    }
  },
  getters: {
    isNewSkill: (state) => {
      return !Object.keys(state.selectedSkill).includes('Id');
    },
    selectedSkillIsEmpty: (state) => {
      return Object.keys(state.selectedSkill?.Skillinfo).length == 0
    }
  },
});

