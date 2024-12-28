<script setup>
import { RouterLink, RouterView } from 'vue-router'
import HelloWorld from './components/HelloWorld.vue'
</script>

<template>
  <header>
    <div class="wrapper">
      <img
        alt="Vue logo"
        class="logo"
        src="@/assets/logo.svg"
        width="100"
        height="100"
      />

      <HelloWorld msg="AI Judge" />
    </div>

    <nav>
      <RouterLink to="/">Home</RouterLink>
      <RouterLink to="/browse">Browse</RouterLink>
      <RouterLink to="/test">Test</RouterLink>
      <RouterLink to="/about">About</RouterLink>
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
  margin: 0 1rem;
}

.wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0.5rem auto;
}

nav {
  width: 100%;
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
  header {
    display:block;
    place-items: center;
    /* padding-right: calc(var(--section-gap) / 2); */
    flex-direction: column;
    height:100vw;
  }

  .logo {
    padding: 0.5rem 0 1rem 0;
  }

  header .wrapper {
    display: flex;
    place-items: center;
    flex-direction: column;
    flex-wrap: wrap;
    width: max-content;
  }

  nav {
    text-align: left;
    margin-left: -1rem;
    font-size: 1rem;

    padding: 1rem;
    margin: 1rem auto 0 auto;

    display: flex;
    flex-direction: column
  }
}
</style>
