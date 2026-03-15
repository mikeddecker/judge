import { createRouter, createWebHistory } from 'vue-router'
import AboutView from '@/views/AboutView.vue'
import HomeView from '@/views/HomeView.vue'
import BrowseView from '@/views/BrowseView.vue'
import ConfigView from '@/views/ConfigView.vue'
import DownloadView from '@/views/DownloadView.vue'
import QuickLabelLocalize from '@/views/QuickLabelLocalize.vue'
import StatsView from '@/views/StatsView.vue'
import TrainView from '@/views/TrainView.vue'
import VideoView from '@/views/VideoView.vue'
import LoginView from '@/views/LoginView.vue'
import RegisterView from '@/views/RegisterView.vue'
import ForgotPasswordView from '@/views/ForgotPasswordView.vue'
import ProfileView from '@/views/ProfileView.vue'
import PermissionsView from '@/views/PermissionsView.vue'
import OpenApiView from '@/views/OpenApiView.vue'
import { useAuthStore } from '@/stores/authStore'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', name: 'home', component: HomeView, meta: { requiresAuth: true } },
    { path: '/about', name: 'about', component: AboutView, meta: { requiresAuth: true } },
    { path: '/browse', name: 'browse', component: BrowseView, meta: { requiresAuth: true } },
    { path: '/config', name: 'config', component: ConfigView, meta: { requiresAuth: true } },
    { path: '/train', name: 'test', component: TrainView, meta: { requiresAuth: true } },
    { path: '/stats', name: 'stats', component: StatsView, meta: { requiresAuth: true } },
    { path: '/video/:id', name: 'video', component: VideoView, meta: { requiresAuth: true } },
    { path: '/download', name: 'download', component: DownloadView, meta: { requiresAuth: true } },
    { path: '/quick-localize', name: 'quick-localize', component: QuickLabelLocalize, meta: { requiresAuth: true } },
    { path: '/login', name: 'login', component: LoginView, meta: { requiresAuth: false } },
    { path: '/register', name: 'register', component: RegisterView, meta: { requiresAuth: false } },
    { path: '/forgot-password', name: 'forgot-password', component: ForgotPasswordView, meta: { requiresAuth: false } },
    { path: '/profile', name: 'profile', component: ProfileView, meta: { requiresAuth: true } },
    { path: '/permissions', name: 'permissions', component: PermissionsView, meta: { requiresAuth: true } },
    { path: '/doc', name: 'doc', component: OpenApiView, meta: { requiresAuth: true } },
  ],
})

// Navigation guard for authentication
router.beforeEach(async (to, from, next) => {
  const authStore = useAuthStore()

  // Initialize auth on first load
  if (!authStore.account && authStore.isAuthenticated === false) {
    await authStore.initializeAuth()
  }

  const requiresAuth = to.meta.requiresAuth ?? true

  if (requiresAuth && !authStore.isAuthenticated) {
    // Redirect to login if trying to access protected route
    next({ name: 'login', query: { redirect: to.fullPath } })
  } else if (!requiresAuth && authStore.isAuthenticated && to.name === 'login') {
    // Redirect to home if already logged in and trying to access login
    next({ name: 'home' })
  } else {
    next()
  }
})

export default router

