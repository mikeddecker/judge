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

export const getSkilloptions = async (skilltype, tablepart) => {
  try {
    const response = await api.get(`/skilloptions/${skilltype}/${tablepart}`)
    return response.data
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
}

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

export const getSkillLevel = async (skillinfo, frameStart, videoId) => {
  try {
    const response = await api.post(`/skilllevel`, { 
        "skillinfo" : skillinfo,
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

export const getStats = async (selectedHar, videoIds) => {
  try {
    return await api.get(
      `/stats`, {
        params: { 'videoIds' : videoIds },
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
