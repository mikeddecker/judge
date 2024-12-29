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

export const getVideoImagePath = async (videoId) => {
  try {
    return await api.get(`/video/${videoId}/image`, { responseType: 'blob' })
      .then(response => {
        let imagePath = URL.createObjectURL(response.data)
        return imagePath
      });
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};