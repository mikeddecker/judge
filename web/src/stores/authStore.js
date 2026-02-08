import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import authService from '@/services/authService'

export const useAuthStore = defineStore('auth', () => {
    const user = ref(null)
    const isAuthenticated = computed(() => user.value !== null)

    // Load user from session on startup
    const initializeAuth = async () => {
      try {
        const data = await authService.me()
        if (data && data.success && data.user) {
          user.value = data.user
        } else {
          user.value = null
        }
        } catch (error) {
            console.error('Failed to fetch user info:', error)
        }
    }

    const setUser = (userData) => {
      user.value = userData
    }

    const clearUser = () => {
      user.value = null
    }

    const logout = async () => {
      try {
        await authService.logout()
      } catch (error) {
        console.error('Logout error:', error)
      } finally {
        clearUser()
      }
    }

    const enableMFA = async () => {
      try {
        const data = await authService.enableMFA()
        if (data && data.success && user.value) {
          user.value.mfaEnabled = true
        }
        return data
      } catch (error) {
        console.error('Enable MFA error:', error)
        throw error
      }
    }

    return {
      user,
      isAuthenticated,
      initializeAuth,
      setUser,
      clearUser,
      logout,
      enableMFA,
    }
})

