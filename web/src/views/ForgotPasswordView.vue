<template>
  <div class="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 py-12 px-4 sm:px-6 lg:px-8">
    <div class="w-full max-w-md space-y-8">
      <div>
        <h2 class="mt-6 text-center text-3xl font-extrabold text-gray-900 dark:text-white">
          Reset your password
        </h2>
      </div>

      <div class="mt-8 bg-white dark:bg-gray-800 py-8 px-6 shadow rounded-lg sm:px-10">
        <!-- Step 1: Request Reset -->
        <form v-if="step === 1" @submit.prevent="handleRequestReset" class="space-y-6">
          <div>
            <label for="email" class="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Email address
            </label>
            <input
              id="email"
              v-model="form.email"
              type="email"
              autocomplete="email"
              required
              class="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 dark:placeholder-gray-500 dark:bg-gray-700 dark:text-white focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              placeholder="you@example.com"
            />
          </div>

          <Message v-if="error" severity="error" class="w-full">{{ error }}</Message>
          <Message v-if="success" severity="success" class="w-full">{{ success }}</Message>

          <Button
            type="submit"
            :loading="loading"
            class="w-full"
            label="Send Reset Link"
          />
        </form>

        <!-- Step 2: Reset Password -->
        <form v-if="step === 2" @submit.prevent="handleResetPassword" class="space-y-6">
          <div>
            <label for="reset-code" class="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Reset Code
            </label>
            <input
              id="reset-code"
              v-model="form.resetCode"
              type="text"
              required
              class="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 dark:placeholder-gray-500 dark:bg-gray-700 dark:text-white focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              placeholder="Code from email"
            />
            <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">
              Check your email for the reset code
            </p>
          </div>

          <div>
            <label for="new-password" class="block text-sm font-medium text-gray-700 dark:text-gray-300">
              New Password
            </label>
            <input
              id="new-password"
              v-model="form.newPassword"
              type="password"
              autocomplete="new-password"
              required
              class="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 dark:placeholder-gray-500 dark:bg-gray-700 dark:text-white focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              placeholder="New Password"
            />
          </div>

          <div>
            <label for="confirm-new-password" class="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Confirm Password
            </label>
            <input
              id="confirm-new-password"
              v-model="form.confirmNewPassword"
              type="password"
              autocomplete="new-password"
              required
              class="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 dark:placeholder-gray-500 dark:bg-gray-700 dark:text-white focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              placeholder="Confirm Password"
            />
          </div>

          <Message v-if="error" severity="error" class="w-full">{{ error }}</Message>
          <Message v-if="success" severity="success" class="w-full">{{ success }}</Message>

          <Button
            type="submit"
            :loading="loading"
            class="w-full"
            label="Reset Password"
          />
        </form>

        <!-- Link to Login -->
        <div class="mt-6 text-center">
          <router-link
            to="/login"
            class="text-sm text-indigo-600 dark:text-indigo-400 hover:text-indigo-500"
          >
            Back to login
          </router-link>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'

import authService from '@/services/authService'

const router = useRouter()

const step = ref(1)
const form = ref({
  email: '',
  resetCode: '',
  newPassword: '',
  confirmNewPassword: '',
})

const loading = ref(false)
const error = ref('')
const success = ref('')

const handleRequestReset = async () => {
    try {
        error.value = ''
        success.value = ''
        loading.value = true

        const data = await authService.forgotPassword(form.value.email)
        console.log('r data', data, data.message)

        success.value = data.message || 'Reset code sent to email'

        // For testing, if a code is returned, use it
        if (data.reset_code) {
            form.value.resetCode = data.reset_code
        }

        setTimeout(() => { step.value = 2 }, 2000)
        } catch (err) {
            error.value = err.response.data.message || 'An error occurred'
        } finally {
            loading.value = false
        }
}

const handleResetPassword = async () => {
  try {
    error.value = ''
    success.value = ''

    if (form.value.newPassword !== form.value.confirmNewPassword) {
      error.value = 'Passwords do not match'
      return
    }

    let min_password_length = process.env.PASSWORD_MIN_LENGTH || 12
    if (form.value.newPassword.length < min_password_length) {
      error.value = `Password must be at least ${min_password_length} characters`
      return
    }

    loading.value = true
    await authService.resetPassword(form.value.resetCode, form.value.newPassword)

    success.value = 'Password reset successfully! Redirecting to login...'
    setTimeout(() => {
      router.push('/login')
    }, 2000)
  } catch (err) {
    error.value = err.response.data.message || 'An error occurred'
    console.error(err)
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
/* Add any additional styling here */
</style>

