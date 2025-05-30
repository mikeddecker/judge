import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:5555', // TODO : process.env.API_URL
  timeout: 300000,
});

export default api;
