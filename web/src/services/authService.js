
import { api } from './api';

const authService = {
  login: async (email, password) => {
    try {
      const resp = await api.post('/auth/login', { email, password });
      return resp.data;
    } catch (err) {
      console.error('login error', err);
      throw err;
    }
  },

  verifyMFA: async (account_id, mfaCode) => {
    try {
      const resp = await api.post('/auth/mfa/verify', { account_id, mfaCode });
      return resp.data;
    } catch (err) {
      console.error('verifyMFA error', err);
      throw err;
    }
  },

  logout: async () => {
    try {
      const resp = await api.post('/auth/logout');
      return resp.data;
    } catch (err) {
      console.error('logout error', err);
      throw err;
    }
  },

  me: async () => {
    try {
      const resp = await api.get('/auth/me');
      return resp.data;
    } catch (err) {
      console.error(err['response'].data.message, err);
      throw err;
    }
  },

  enableMFA: async () => {
    try {
      const resp = await api.post('/auth/enable-mfa');
      return resp.data;
    } catch (err) {
      console.error('enableMFA error', err);
      throw err;
    }
  },

  forgotPassword: async (email) => {
    try {
      const resp = await api.post('/auth/forgot-password', { email });
      return resp.data;
    } catch (err) {
      console.error('forgotPassword error', err);
      throw err;
    }
  },

  resetPassword: async (token, newPassword) => {
    try {
      const resp = await api.post('/auth/reset-password', { token, newPassword });
      return resp.data;
    } catch (err) {
      console.error('resetPassword error', err);
      throw err;
    }
  },

  register: async (payload) => {
    try {
      const resp = await api.post('/auth/register', payload);
      return resp.data;
    } catch (err) {
      console.error('register error', err);
      throw err;
    }
  },
};

export default authService;

