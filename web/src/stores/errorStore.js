import { defineStore } from "pinia";

export const useErrorStore = defineStore("error", {
  state: () => {
    return {
      error: null,
    };
  },
  actions: {
    setError(message) {
      this.error = message;
      setTimeout(() => { this.error = null; }, 4000);
    },
  },
});

