import api from './api';

export const getFolder = async (folderId) => {
  try {
    const response = await api.get(`/folders/${folderId}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};

export const getVideoInfo = async (videoId) => {
  try {
    const response = await api.get(`/video/${videoId}/info`);
    return response.data;
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};

export const getVideoImagePath = async (videoId) => {
  try {
    return await api.get(`/video/${videoId}/image`, { responseType: 'blob', timeout: 30000 })
      .then(response => {
        let imagePath = URL.createObjectURL(response.data)
        return imagePath
      });
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};

export const createVideoImage = async (videoId, frameNr) => {
  return await api.post(`/video/${videoId}/image`, frameNr, { headers: { 'Content-Type': 'application/json' }})
  .then(function (response) {
    return response;
  })
  .catch(function (error) {
    console.error(error);
  });
};

export const getVideoPath = async (videoId) => {
  try {
    return await api.get(`/video/${videoId}`, { responseType: 'blob' })
      .then(response => {
        let videoPath = URL.createObjectURL(response.data)
        return videoPath
      });
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};

export const getCroppedVideoPath = async (videoId) => {
  try {
    return await api.get(`/video/${videoId}/cropped`, { responseType: 'blob' })
      .then(response => {
        let videoPath = URL.createObjectURL(response.data)
        return videoPath
      });
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};

export const postVideoFrame = async (videoId, frameNr, frameinfo) => {
  return await api.post(`/video/${videoId}/frameNr/${frameNr}`, frameinfo, { headers: { 'Content-Type': 'application/json' }})
    .then(function (response) {
      return response.data;
    })
    .catch(function (error) {
      console.error(error);
    });
};

export const removeVideoFrame = async (videoId, frameNr, frameinfo) => {
  return await api.delete(`/video/${videoId}/frameNr/${frameNr}`, { 
      headers: { 'Content-Type': 'application/json' },
      data: { frameinfo },
    })
    .then(function (response) {
      return response.data;
    })
    .catch(function (error) {
      console.error(error);
    });
};

export const downloadVideo = async (downloadinfo) => {
  return await api.post(`/download`, downloadinfo, { headers: { 'Content-Type': 'application/json' }})
    .catch(function (error) {
      throw error;
    });
};

export const postSkill = async (videoId, skillinfo) => {
  return await api.post(`/skill/${videoId}`, skillinfo, { headers: { 'Content-Type': 'application/json' }})
    .then(function (response) {
      return response.data;
    })
    .catch(function (error) {
      console.error(error);
    });
};

export const putSkill = async (videoId, skillinfo) => {
  return await api.put(`/skill/${videoId}`, skillinfo, { headers: { 'Content-Type': 'application/json' }})
    .then(function (response) {
      return response.data;
    })
    .catch(function (error) {
      console.error(error);
    });
};

export const deleteSkill = async (videoId, start, end) => {
  return await api.delete(`/skill/${videoId}`, { 
      headers: { 'Content-Type': 'application/json' },
      data: { "FrameStart": start, "FrameEnd": end },
    })
    .then(function (response) {
      return response.data;
    })
    .catch(function (error) {
      console.error(error);
    });
};

export const getSkillLevel = async (skillinfo, prevSkillinfo, prevSkillname, frameStart, videoId) => {
  try {
    const response = await api.post(`/skilllevel`, { 
        "skillinfo" : skillinfo,
        "prevSkillinfo" : {"Skillinfo": prevSkillinfo},
        "prevSkillname" : prevSkillname,
        "frameStart" : frameStart,
        "videoId" : videoId,
      }, { headers: { 'Content-Type': 'application/json' }})
    return response.data
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
}

export const updateVideoSkillsCompleted = async (videoId, completed) => {
  try {
    const response = await api.post(
      `/skillcompleted/${videoId}`, 
      { 
        "completed" : completed,
      }, 
      { headers: { 'Content-Type': 'application/json' }}
    )
    return response.data
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
}

export const getStats = async (stat, videoIds) => {
  try {
    return await api.get(
      `/stats`, {
        params: { 'stat': stat, 'videoIds' : videoIds },
        headers: { 'Content-Type': 'application/json' }
      }).then(response => response.data)
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};

export const getLocalizeStats = async (selectedHar) => {
  try {
    return await api.get(`/stats/localize`).then(response => response.data)
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};

export const getVideoPredictions = async (videoId) => {
  try {
    return await api.get(`/video/${videoId}/predictions`).then(response => response.data)
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};

export const hasLocalizePredictions = async (videoId) => {
  try {
    return await api.get(`/video/${videoId}/predictions/hasLocalizePredictions`).then(response => response.data)
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};
export const getLocalizePredictions = async (videoId) => {
  try {
    return await api.get(`/video/${videoId}/predictions/getLocalizePredictions`).then(response => response.data)
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};

export const launchJob = async (jobarguments) => {
  try {
    return await api.post('/job', jobarguments, { headers: { 'Content-Type': 'application/json' }}).then(response => response.data)
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};

export const discoverDrive = async () => {
  try {
    return await api.get(`/discover/deleteOrphans`)
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};

export const getTags = async () => {
  try {
    return await api.get(`/tags`).then(response => response.data)
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};

export const getTagGroups = async () => {
  try {
    return await api.get(`/tagGroups`).then(response => response.data)
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};

export const addTag = async (name, group) => {
  return await api.post('/tags', { 'name': name, 'group': group }, { headers: { 'Content-Type': 'application/json' }})
  .then(function (response) {
    return response;
  })
  .catch(function (error) {
    console.error(error);
  });
};

export const addTagGroup = async (name) => {
  return await api.post('/tagGroups', { 'name': name }, { headers: { 'Content-Type': 'application/json' }})
  .then(function (response) {
    return response;
  })
  .catch(function (error) {
    console.error(error);
  });
};

export const updateTag = async (id, name, keywords) => {
  return await api.put('/tags', { 'id': id, 'name': name, 'keywords': keywords}, { headers: { 'Content-Type': 'application/json' }})
  .then(function (response) {
    return response;
  })
  .catch(function (error) {
    console.error(error);
  });
};

export const updateTagGroup = async (id, group) => {
  return await api.put('/tags', { 'id': id, 'group': group }, { headers: { 'Content-Type': 'application/json' }})
  .then(function (response) {
    return response;
  })
  .catch(function (error) {
    console.error(error);
  });
};

export const getFrameLabelTypes = async () => {
  try {
    return await api.get('/frameLabelTypes').then(response => response.data)
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
}

export const getJobOptions = async (step) => {
  try {
    return await api.get(`/job/options/${step}`).then(response => response.data)
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
}

export const getLayers = async () => {
  try {
    return await api.get(`/layers`).then(response => response.data)
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};

export const getLayerTypes = async () => {
  try {
    return await api.get(`/layers/types`).then(response => response.data)
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};

export const moveLayer = async (compositionName, key, sourceStage, destStage, stageNr) => {
  return await api.post('/layers/move', { compositionName, key, sourceStage, destStage, stageNr }, { headers: { 'Content-Type': 'application/json' }})
  .then(function (response) {
    return response.data;
  })
  .catch(function (error) {
    console.error(error);
  });
};

export const addLayer = async (name, layerId, type, min, max, step) => {
  return await api.post('/layers', { name, layerId, type, min, max, step }, { headers: { 'Content-Type': 'application/json' }})
  .then(function (response) {
    return response.data;
  })
  .catch(function (error) {
    console.error(error);
  });
};

export const updateLayer = async (id, name, layerId, min, max, step) => {
  return await api.post('/layers', { id, name, layerId, min, max, step }, { headers: { 'Content-Type': 'application/json' }})
  .then(function (response) {
    return response.data;
  })
  .catch(function (error) {
    console.error(error);
  });
};

export const getLayerCompositions = async () => {
  try {
    return await api.get(`/layercompositions`).then(response => response.data)
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};

export const addLayerComposition = async (compositionName, stage, layerId, name) => {
  return await api.post('/layercompositions', { compositionName, stage, layerId, name }, { headers: { 'Content-Type': 'application/json' }})
  .then(function (response) {
    return response.data;
  })
  .catch(function (error) {
    console.error(error);
  });
};

export const updateLayerCompositionAttributeValue = async (compositionName, stage, layername, attribute, value) => {
  return await api.post('/layercompositions/attribute', { compositionName, stage, layername, attribute, value }, { headers: { 'Content-Type': 'application/json' }})
  .then(function (response) {
    return response.data;
  })
  .catch(function (error) {
    console.error(error);
  });
};

