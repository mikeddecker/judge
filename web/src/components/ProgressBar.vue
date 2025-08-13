<template>
  <div class="progressbar">
    <div :style="fillerStyles">
      <span class="label">{{ `${percentageCompleted}%` }}</span>
    </div>
  </div>
</template>
  
<script>
import { computed } from "vue";
export default {
  name: "ProgressBar",
  props: {
    bgcolor: String,
    percentageCompleted: Number,
  },
  setup(props) {
    const fillerStyles = computed(() => {
      return {
        height: "100%",
        width: `${Math.min(100, props.percentageCompleted)}%`,
        backgroundColor: props.bgcolor,
        transition: "width 1s ease-in-out",
        borderRadius: "inherit",
        textAlign: "right",
      };
    });

    return { fillerStyles };
  },
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.progressbar {
  width: 95%;
  background-color: #e0e0de;
  border-radius: 50px;
  position: absolute;
  bottom: 0.25rem;
}

.label {
  padding: 5px;
  color: white;
  font-weight: bold;
}
</style>

