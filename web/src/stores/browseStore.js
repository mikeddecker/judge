import { defineStore } from "pinia";

export const useBrowseStore = defineStore("browse", {
  state: () => {
    return {
      lastVisitedFolder: null,
    };
  },
  actions: {
    setLastVisitedFolder(folderId) {
      this.lastVisitedFolder = folderId;
    },
  },
});

