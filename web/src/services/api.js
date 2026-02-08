import axios from 'axios';

export const api = axios.create({
  baseURL: '/api',
  timeout: 300000,
  withCredentials: true, // Enable cookies for session management
});

export const getApplicationJson = async (route, params) => {
  try {
    const response = await api.get(route, {
      params: params,
      headers: { 'Content-Type': 'application/json' },
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};

export default api;

