<template>
  <div>
    <h1>API Data</h1>
    <div v-if="loading">Loading...</div>
    <div v-if="error" class="error">{{ error }}</div>
    <div v-if="data">
      <pre>{{ data }}</pre>
    </div>
  </div>
</template>

<script>
import { getFolder } from '../services/videoService';

export default {
  data() {
    return {
      data: null,
      loading: false,
      error: null,
    };
  },
  async created() {
    this.loading = true;
    try {
      this.data = await getFolder(0);
    } catch {
      this.error = 'Failed To load';
    } finally {
      this.loading = false;
    }
  },
};
</script>

<style scoped>
.error {
  color: red;
}
</style>
