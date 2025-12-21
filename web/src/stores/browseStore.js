import { defineStore } from "pinia";

export const useBrowseStore = defineStore("browse", {
  state: () => {
    return {
      lastVisitedFolder: 0,
    };
  },
  actions: {
    setLastVisitedFolder(folderId) {
      this.lastVisitedFolder = folderId;
    },
  },
});

