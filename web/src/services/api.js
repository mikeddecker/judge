import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:5555', // Your backend API URL
  timeout: 10000, // Optional timeout
});

export default api;
