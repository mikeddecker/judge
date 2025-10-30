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
      const getStageFocussedDefaultValues = (layerproperty) => {
        // Fills in stage with default values 
        let stageDefaultValues = {}
        Object.entries(layerproperty).forEach(([propertyname, propertyinfo]) => {
          if (propertyinfo.focussed) {
            stageDefaultValues[propertyname] = propertyinfo.defaultValue
          }
        })
        return stageDefaultValues
      }

      let includeDefaultValues = false // TODO : add toggle in config.

      let selectedLayerComposition = this.layercomposition[selectedCompositionName]
      let label = !includeDefaultValues ? {} : Object.fromEntries(
        Object.entries(selectedLayerComposition).map(([upperStage, upperStageValues]) => {
          if (upperStage == 'compositionName') { return [ upperStage, upperStageValues ]}
          if (upperStage == 'StageProperties') {
            return [upperStage, Object.fromEntries(Object.entries(upperStageValues).map(([stageNr, stageValues]) => {
                return [stageNr, getStageFocussedDefaultValues(stageValues)]
            }))]
          } else {
            return [upperStage, getStageFocussedDefaultValues(upperStageValues)]
          }
        }
      ));
      
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
      this.selectedSkill.Skillinfo[compositionName].splice(index, 1)
      if (!this.selectedSkill.Skillinfo[compositionName].length) {
        delete this.selectedSkill.Skillinfo[compositionName]
      }
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

