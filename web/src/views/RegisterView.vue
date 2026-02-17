<template>
  <div class="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 py-12 px-4 sm:px-6 lg:px-8">
    <div class="w-full max-w-md space-y-8">
      <div>
        <h2 class="mt-6 text-center text-3xl font-extrabold text-gray-900 dark:text-white">
          Create your account
        </h2>
      </div>

      <div class="mt-8 bg-white dark:bg-gray-800 py-8 px-6 shadow rounded-lg sm:px-10">
        <form @submit.prevent="handleRegister" class="space-y-6">
          <div>
            <label for="first-name" class="block text-sm font-medium text-gray-700 dark:text-gray-300">
              First Name
            </label>
            <input
              id="first-name"
              v-model="form.firstName"
              type="text"
              required
              class="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 dark:placeholder-gray-500 dark:bg-gray-700 dark:text-white focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              placeholder="John"
            />
          </div>

          <div>
            <label for="last-name" class="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Last Name
            </label>
            <input
              id="last-name"
              v-model="form.lastName"
              type="text"
              required
              class="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 dark:placeholder-gray-500 dark:bg-gray-700 dark:text-white focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              placeholder="Doe"
            />
          </div>

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
              autocomplete="new-password"
              required
              class="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 dark:placeholder-gray-500 dark:bg-gray-700 dark:text-white focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              :placeholder="`Password (minimum ${passwordMinLength} characters)`"
            />
            <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">
              Must be at least {{ passwordMinLength }} characters long
            </p>
          </div>

          <div>
            <label for="confirm-password" class="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Confirm Password
            </label>
            <input
              id="confirm-password"
              v-model="form.confirmPassword"
              type="password"
              autocomplete="new-password"
              required
              class="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 dark:placeholder-gray-500 dark:bg-gray-700 dark:text-white focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              placeholder="Confirm Password"
            />
          </div>

          <!-- Error Message -->
          <Message v-if="error" severity="error" class="w-full">{{ error }}</Message>

          <!-- Success Message -->
          <Message v-if="success" severity="success" class="w-full">{{ success }}</Message>

          <!-- Loading State -->
          <Button
            type="submit"
            :loading="loading"
            class="w-full"
            label="Create Account"
          />
        </form>

        <!-- Link to Login -->
        <div class="mt-6 text-center">
          <router-link
            to="/login"
            class="text-sm text-indigo-600 dark:text-indigo-400 hover:text-indigo-500"
          >
            Already have an account? Sign in
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
import Button from 'primevue/button'
import Message from 'primevue/message'

const router = useRouter()

const form = ref({
  firstName: '',
  lastName: '',
  email: '',
  password: '',
  confirmPassword: '',
})

const loading = ref(false)
const error = ref('')
const success = ref('')

const passwordMinLength = process.env.PASSWORD_MIN_LENGTH;


const handleRegister = async () => {
  try {
    error.value = ''
    success.value = ''

    // Validation
    if (form.value.password !== form.value.confirmPassword) {
      error.value = 'Passwords do not match'
      return
    }

    console.log('huh', passwordMinLength, form.value.password.length < passwordMinLength)
    if (form.value.password.length < process.env.PASSWORD_MIN_LENGTH) {
      error.value = `Password must be at least ${passwordMinLength} characters`
      return
    }

    loading.value = true

    const data = await authService.register({
      firstName: form.value.firstName,
      lastName: form.value.lastName,
      email: form.value.email,
      password: form.value.password,
    })

    if (!data || !data.success) {
      error.value = data?.message || 'Registration failed'
      return
    }

    success.value = 'Account created successfully! Redirecting to login...'
    setTimeout(() => {
      router.push('/login')
    }, 2000)
  } catch (err) {
    error.value = err.response.data.message || 'An error occurred'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
/* Add any additional styling here */
</style>

