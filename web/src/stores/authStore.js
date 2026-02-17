import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import authService from '@/services/authService'

export const useAuthStore = defineStore('auth', () => {
    const account = ref(null)
    const isAuthenticated = computed(() => account.value !== null)

    // Load account from session on startup
    const initializeAuth = async () => {
      try {
        const data = await authService.me()
        if (data && data.success && data.account) {
          account.value = data.account
        } else {
          account.value = null
        }
        } catch (error) {
            console.error('Failed to fetch account info:', error)
        }
    }

    const setAccount = (accountData) => {
      account.value = accountData
    }

    const clearAccount = () => {
      account.value = null
    }

    const logout = async () => {
      try {
        await authService.logout()
      } catch (error) {
        console.error('Logout error:', error)
      } finally {
        clearAccount()
      }
    }

    const enableMFA = async () => {
      try {
        const data = await authService.enableMFA()
        if (data && data.success && account.value) {
          account.value.mfaEnabled = true
        }
        return data
      } catch (error) {
        console.error('Enable MFA error:', error)
        throw error
      }
    }

    return {
      account,
      isAuthenticated,
      initializeAuth,
      setAccount,
      clearAccount,
      logout,
      enableMFA,
    }
})

