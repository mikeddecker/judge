<template>
  <div class="profile-container">
    <h1>Profile</h1>

    <div class="profile-sections">
      <div class="profile-section">
        <h2>Account</h2>
        <p class="section-desc">Manage your account details, password, and MFA settings.</p>
        <p class="text-sm text-gray-500 dark:text-gray-400">Coming soon...</p>
      </div>

      <div class="profile-section">
        <h2>Permissions</h2>
        <p class="section-desc">Manage groups, access grants, and blocked accounts.</p>
        <RouterLink to="/permissions" class="profile-link">
          Go to Permissions →
        </RouterLink>
      </div>
    </div>
  </div>
</template>

<script setup>
import { RouterLink } from 'vue-router'
</script>

<style scoped>
.profile-container {
  padding: 2rem;
  max-width: 700px;
  margin: 0 auto;
}

h1 {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 1.5rem;
}

.profile-sections {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

.profile-section {
  border: 1px solid var(--color-border, #e5e7eb);
  border-radius: 0.5rem;
  padding: 1.25rem;
}

h2 {
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 0.35rem;
}

.section-desc {
  font-size: 0.875rem;
  color: #6b7280;
  margin-bottom: 0.75rem;
}

.profile-link {
  font-size: 0.875rem;
  color: var(--color-link, #4f46e5);
  text-decoration: none;
  font-weight: 500;
}

.profile-link:hover {
  text-decoration: underline;
}
</style>

