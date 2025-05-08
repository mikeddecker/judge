import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '@/views/HomeView.vue'
import TestView from '@/views/TestView.vue'
import VideoView from '@/views/VideoView.vue'
import AboutView from '@/views/AboutView.vue'
import BrowseView from '@/views/BrowseView.vue'
import DownloadView from '@/views/DownloadView.vue'
import StatsView from '@/views/StatsView.vue'
import QuickLabelLocalize from '@/views/QuickLabelLocalize.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', name: 'home', component: HomeView },
    { path: '/about', name: 'about', component: AboutView, },
    { path: '/browse', name: 'browse', component: BrowseView },
    { path: '/test', name: 'test', component: TestView },
    { path: '/stats', name: 'stats', component: StatsView },
    { path: '/video/:id', name: 'video', component: VideoView },
    { path: '/download', name: 'download', component: DownloadView },
    { path: '/quick-localize', name: 'quick-localize', component: QuickLabelLocalize },
  ],
})

export default router
