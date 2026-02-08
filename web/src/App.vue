<script setup>
import { RouterLink, RouterView, useRoute, useRouter } from 'vue-router'
import HelloWorld from './components/HelloWorld.vue'
import { computed, onMounted } from 'vue'
import { useAuthStore } from '@/stores/authStore'
import { ref } from 'vue'

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()

const displayLogo = computed(() => route.name ? ['browse', 'about', 'stats', 'home', 'profile'].includes(route.name) : true)

const menuItems = ref([
  {
    label: 'Profile',
    icon: 'pi pi-cog',
    command: () => {
      router.push('/profile')
    }
  },
  // MFA disabled for now
  // {
  //   label: 'Enable MFA',
  //   icon: 'pi pi-shield',
  //   command: () => {
  //     authStore.enableMFA()
  //   }
  // },
  {
    separator: true
  },
  {
    label: 'Logout',
    icon: 'pi pi-sign-out',
    command: () => {
      authStore.logout()
      location.href = '/login'
    }
  }
])

const userMenu = ref()

onMounted(async () => {
  await authStore.initializeAuth()
})

const toggleUserMenu = (event) => {
  userMenu.value.toggle(event)
}
</script>

<template>
  <header>
    <div class="wrapper flex justify-between items-center" v-if="displayLogo">
      <div class="flex items-center">
        <img
          alt="Vue logo"
          class="logo"
          src="@/assets/logo.svg"
          width="50"
          height="50"
        />
        <HelloWorld msg="AI Judge" />
      </div>

      <!-- User Menu -->
      <div v-if="authStore.isAuthenticated" class="flex items-center gap-3 ml-2">
        <!-- <span class="text-sm text-gray-600">{{ authStore.user?.firstName }} {{ authStore.user?.lastName }}</span> -->
        <Button
          @click="toggleUserMenu"
          icon="pi pi-user"
          rounded
          severity="secondary"
        />
        <Menu ref="userMenu" :model="menuItems" :popup="true" />
      </div>
    </div>

    <nav>
      <RouterLink v-if="authStore.isAuthenticated" to="/">Home</RouterLink>
      <RouterLink v-if="authStore.isAuthenticated" to="/browse">Browse</RouterLink>
      <RouterLink v-if="authStore.isAuthenticated" to="/train">Train</RouterLink>
      <RouterLink v-if="authStore.isAuthenticated" to="/stats">Stats</RouterLink>
      <RouterLink v-if="authStore.isAuthenticated" to="/about">About</RouterLink>
      <RouterLink v-if="authStore.isAuthenticated" to="/config">Config</RouterLink>
      <RouterLink v-if="!authStore.isAuthenticated" to="/login">Login</RouterLink>
      <RouterLink v-if="!authStore.isAuthenticated" to="/register">Register</RouterLink>
    </nav>
  </header>
  <main class="mb-32">
    <RouterView/>
  </main>
</template>

<style scoped>
header {
  line-height: 1;
  max-height: 100vh;
  background-color: var(--color-nav);
  padding: 0.5rem;
}

.logo {
  display: block;
}

.wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0.5rem auto;
}

nav {
  font-size: 12px;
  text-align: center;
  padding-top: 1rem;
}

nav a.router-link-exact-active {
  color: var(--color-text);
}

nav a.router-link-exact-active:hover {
  background-color: transparent;
}

nav a {
  display: inline-block;
  padding: 0 1rem;
  border-left: 1px solid var(--color-border);
}

nav a:first-of-type {
  border: 0;
}

@media (min-width: 1024px) {
}
</style>

