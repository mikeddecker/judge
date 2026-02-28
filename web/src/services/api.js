import { useErrorStore } from '@/stores/errorStore';
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
    const errorStore = useErrorStore()
    errorStore.setError((error.response?.data || error.response?.data?.message) ?? 'Something went wrong');
  }
};

export const postApplicationJson = async (route, data) => {
  return await api.post(route, data, { headers: { 'Content-Type': 'application/json' }})
    .then(function (response) {
      return response.data;
    })
    .catch(function (error) {
      const errorStore = useErrorStore()
      errorStore.setError((error.response?.data || error.response?.data?.message) ?? 'Something went wrong');
    });
};

export default api;

