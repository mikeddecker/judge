<template>
  <div class="p-8 max-w-2xl mx-auto">
    <h1 class="text-2xl font-bold mb-6">Profile</h1>

    <div class="flex flex-col gap-5">
      <div class="border border-gray-200 dark:border-gray-700 rounded-lg p-5">
        <h2 class="text-base font-semibold mb-1">Account</h2>
        <p class="text-sm text-gray-500 dark:text-gray-400 mb-3">Manage your account details, password, and MFA settings.</p>
        <p class="text-sm text-gray-400 dark:text-gray-500">Coming soon...</p>
      </div>

      <div class="border border-gray-200 dark:border-gray-700 rounded-lg p-5">
        <h2 class="text-base font-semibold mb-1">Permissions</h2>
        <p class="text-sm text-gray-500 dark:text-gray-400 mb-3">Manage groups, access grants, and blocked accounts.</p>
        <RouterLink to="/permissions" class="text-sm font-medium text-indigo-600 dark:text-indigo-400 hover:underline">
          Go to Permissions →
        </RouterLink>
      </div>
    </div>
  </div>
</template>

<script setup>
import { RouterLink } from 'vue-router'
</script>

