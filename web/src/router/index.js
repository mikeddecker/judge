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

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', name: 'home', component: HomeView },
    { path: '/about', name: 'about', component: AboutView, },
    { path: '/browse', name: 'browse', component: BrowseView },    
    { path: '/config', name: 'config', component: ConfigView, },
    { path: '/train', name: 'test', component: TrainView },
    { path: '/stats', name: 'stats', component: StatsView },
    { path: '/video/:id', name: 'video', component: VideoView },
    { path: '/download', name: 'download', component: DownloadView },
    { path: '/quick-localize', name: 'quick-localize', component: QuickLabelLocalize },
  ],
})

export default router
