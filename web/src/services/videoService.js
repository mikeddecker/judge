import api from './api';

export const getFoldersRoot = async () => {
  try {
    const response = await api.get('/folders');
    return response.data;
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};
