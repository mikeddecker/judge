<script setup>
import { RouterLink, RouterView, useRoute } from 'vue-router'
import HelloWorld from './components/HelloWorld.vue'
import { computed } from 'vue';

const route = useRoute()

const displayLogo = computed(() => route.name ? ['browse', 'about', 'stats', 'home'].includes(route.name) : true)

</script>

<template>
  <header>
    <div class="wrapper" v-if="displayLogo">
      <img
        alt="Vue logo"
        class="logo"
        src="@/assets/logo.svg"
        width="50"
        height="50"
      />
      <HelloWorld msg="AI Judge" />
    </div>
    
    <nav>
      <RouterLink to="/">Home</RouterLink>
      <RouterLink to="/browse">Browse</RouterLink>
      <RouterLink to="/download">Download</RouterLink>
      <RouterLink to="/test">Test</RouterLink>
      <RouterLink to="/stats">Stats</RouterLink>
      <RouterLink to="/about">About</RouterLink>
      <!-- <RouterLink to="/quick-localize">QuickLocalize</RouterLink> -->
    </nav>
  </header>
  <main>
    <RouterView />
  </main>
</template>

<style scoped>
header {
  line-height: 1;
  max-height: 100vh;
  background-color: var(--color-nav);
  padding: 1rem;
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
