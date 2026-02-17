<template>
  <div class="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 py-12 px-4 sm:px-6 lg:px-8">
    <div class="w-full max-w-md space-y-8">
      <div>
        <h2 class="mt-6 text-center text-3xl font-extrabold text-gray-900 dark:text-white">
          Sign in to your account
        </h2>
      </div>

      <!-- Login Form -->
      <div v-if="!showMFA" class="mt-8 bg-white dark:bg-gray-800 py-8 px-6 shadow rounded-lg sm:px-10">
        <form @submit.prevent="handleLogin" class="space-y-6">
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

          <div>
            <label for="password" class="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Password
            </label>
            <input
              id="password"
              v-model="form.password"
              type="password"
              autocomplete="current-password"
              required
              class="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 dark:placeholder-gray-500 dark:bg-gray-700 dark:text-white focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              placeholder="Password"
            />
          </div>

          <!-- Error Message -->
          <Message v-if="error" severity="error" class="w-full">{{ error }}</Message>

          <!-- Loading State -->
          <Button
            type="submit"
            :loading="loading"
            class="w-full"
            label="Sign in"
          />
        </form>

        <!-- Links -->
        <div class="mt-6 flex items-center justify-between">
          <router-link
            to="/forgot-password"
            class="text-sm text-indigo-600 dark:text-indigo-400 hover:text-indigo-500"
          >
            Forgot your password?
          </router-link>
          <router-link
            to="/register"
            class="text-sm text-indigo-600 dark:text-indigo-400 hover:text-indigo-500"
          >
            Create account
          </router-link>
        </div>
      </div>

      <!-- MFA Verification Form -->
      <div v-if="showMFA" class="mt-8 bg-white dark:bg-gray-800 py-8 px-6 shadow rounded-lg sm:px-10">
        <p class="text-center text-sm text-gray-600 dark:text-gray-400 mb-6">
          A verification code has been sent to your email
        </p>

        <form @submit.prevent="handleMFAVerify" class="space-y-6">
          <div>
            <label for="mfa-code" class="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Verification Code
            </label>
            <input
              id="mfa-code"
              v-model="form.mfaCode"
              type="text"
              inputmode="numeric"
              maxlength="6"
              required
              class="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 dark:placeholder-gray-500 dark:bg-gray-700 dark:text-white focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm text-center text-2xl tracking-widest"
              placeholder="000000"
            />
            <p class="mt-2 text-xs text-gray-500 dark:text-gray-400">
              Enter the 6-digit code from your email
            </p>
          </div>

          <!-- Error Message -->
          <Message v-if="error" severity="error" class="w-full">{{ error }}</Message>

          <!-- Loading State -->
          <Button
            type="submit"
            :loading="loading"
            class="w-full"
            label="Verify"
          />
        </form>

        <Button
          text
          @click="showMFA = false"
          class="w-full mt-4"
          label="Back to login"
          severity="secondary"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/authStore'
import authService from '@/services/authService'
import Button from 'primevue/button'
import Message from 'primevue/message'

const router = useRouter()
const authStore = useAuthStore()

const form = ref({
  email: '',
  password: '',
  mfaCode: '',
})

const loading = ref(false)
const error = ref('')
const showMFA = ref(false)
const accountId = ref(null)

const handleLogin = async () => {
  try {
    loading.value = true
    error.value = ''

    const data = await authService.login(form.value.email, form.value.password)

    if (!data) {
      error.value = 'Login failed'
      return
    }

    if (data.requires_mfa) {
      accountId.value = data.account_id
      showMFA.value = true
      form.value.mfaCode = ''
    } else {
      // Store account info and redirect
      authStore.setAccount(data.account)
      await router.push('/')
    }
  } catch (err) {
    error.value = err.response.data.message || 'An error occurred'
  } finally {
    loading.value = false
  }
}

const handleMFAVerify = async () => {
  try {
    loading.value = true
    error.value = ''

    const data = await authService.verifyMFA(accountId.value, form.value.mfaCode)

    if (!data) {
      error.value = 'Verification failed'
      return
    }

    authStore.setAccount(data.account)
    await router.push('/')
  } catch (err) {
    error.value = err.response.data.message || 'An error occurred'
  } finally {
    loading.value = false
  }
}
</script>

