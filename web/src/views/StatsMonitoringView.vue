<template>
  <div class="monitoring-dashboard">
    <!-- Admin-only warning -->
    <div v-if="!isAdmin" class="mb-4 p-3 bg-blue-50 border border-blue-200 rounded text-sm text-blue-700">
      ℹ️ Showing limited metrics. Full monitoring available to administrators.
    </div>

    <div class="grid grid-cols-2 gap-4 mb-6">
      <!-- Queue Depth Card (Admin only) -->
      <div v-if="isAdmin" class="card p-4 rounded border border-gray-300">
        <h3 class="font-bold mb-3 text-red-600">Job Queue</h3>
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span>🤖 AI Jobs Pending:</span>
            <span class="font-mono">{{ monitoring?.queue?.ai_jobs_pending || 0 }}</span>
          </div>
          <div class="flex justify-between">
            <span>🔄 Sync Jobs Pending:</span>
            <span class="font-mono">{{ monitoring?.queue?.sync_jobs_pending || 0 }}</span>
          </div>
          <div class="flex justify-between">
            <span>💾 Backup Jobs Pending:</span>
            <span class="font-mono">{{ monitoring?.queue?.backup_jobs_pending || 0 }}</span>
          </div>
          <hr class="my-2" />
          <div class="flex justify-between font-bold">
            <span>Total Pending:</span>
            <span class="font-mono">{{ monitoring?.queue?.total_pending || 0 }}</span>
          </div>
          <div class="flex justify-between text-xs text-gray-600 mt-2">
            <span>✅ Completed:</span>
            <span>{{ monitoring?.queue?.jobs_completed_total || 0 }}</span>
          </div>
          <div class="flex justify-between text-xs text-red-600">
            <span>❌ Failed:</span>
            <span>{{ monitoring?.queue?.jobs_failed_total || 0 }}</span>
          </div>
        </div>
      </div>

      <!-- Conflicts Card (Admin only) -->
      <div v-if="isAdmin" class="card p-4 rounded border border-gray-300">
        <h3 class="font-bold mb-3 text-orange-600">Conflicts</h3>
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span>⚠️ Unresolved:</span>
            <span class="font-mono text-red-600 font-bold">{{ monitoring?.conflicts?.unresolved_count || 0 }}</span>
          </div>
          <div class="flex justify-between">
            <span>✅ Auto-Resolved:</span>
            <span class="font-mono text-green-600">{{ monitoring?.conflicts?.auto_resolved_count || 0 }}</span>
          </div>
          <hr class="my-2" />
          <div class="flex justify-between font-bold">
            <span>Total Conflicts:</span>
            <span class="font-mono">{{ monitoring?.conflicts?.total_conflicts || 0 }}</span>
          </div>
        </div>
      </div>

      <!-- Data Metrics Card (Visible to all) -->
      <div class="card p-4 rounded border border-gray-300">
        <h3 class="font-bold mb-3 text-blue-600">Data Metrics</h3>
        <div class="space-y-2 text-sm">
          <div v-if="isAdmin" class="flex justify-between">
            <span>📹 Total Videos:</span>
            <span class="font-mono">{{ monitoring?.data?.total_videos || 0 }}</span>
          </div>
          <div class="flex justify-between">
            <span>✏️ Labeled Videos:</span>
            <span class="font-mono">{{ monitoring?.data?.labeled_videos || 0 }}</span>
          </div>
          <div class="flex justify-between">
            <span>📊 Labeling %:</span>
            <span class="font-mono font-bold">{{ monitoring?.data?.labeling_percentage || 0 }}%</span>
          </div>
          <div v-if="isAdmin" class="border-t pt-2 flex justify-between text-xs text-gray-600">
            <span>🗑 Soft Deleted:</span>
            <span class="font-mono">{{ monitoring?.data?.soft_deleted_videos || 0 }}</span>
          </div>
        </div>
      </div>

      <!-- Storage Card (Admin only) -->
      <div v-if="isAdmin" class="card p-4 rounded border border-gray-300">
        <h3 class="font-bold mb-3 text-purple-600">Storage Usage</h3>
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span>📹 Videos:</span>
            <span class="font-mono">{{ monitoring?.storage?.videos_gb || 0 }} GB</span>
          </div>
          <div class="flex justify-between">
            <span>📊 Generated Data:</span>
            <span class="font-mono">{{ monitoring?.storage?.generated_data_gb || 0 }} GB</span>
          </div>
          <hr class="my-2" />
          <div class="flex justify-between font-bold">
            <span>Total Storage:</span>
            <span class="font-mono text-lg">{{ monitoring?.storage?.total_gb || 0 }} GB</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Region and Last Update -->
    <div class="text-xs text-gray-500 mt-4 p-2 bg-gray-100 rounded">
      <div>Region: <span class="font-mono">{{ monitoring?.region || 'unknown' }}</span></div>
      <div>Last Updated: <span class="font-mono">{{ formatTime(monitoring?.timestamp) }}</span></div>
    </div>
  </div>
</template>

<script setup>
import { getStats } from '../services/videoService';
import { onMounted, ref } from 'vue';

const isAdmin = ref(false);

const monitoring = ref({
  queue: {
    ai_jobs_pending: 0,
    sync_jobs_pending: 0,
    backup_jobs_pending: 0,
    total_pending: 0,
    jobs_completed_total: 0,
    jobs_failed_total: 0
  },
  conflicts: {
    unresolved_count: 0,
    auto_resolved_count: 0,
    total_conflicts: 0
  },
  data: {
    total_videos: 0,
    labeled_videos: 0,
    labeling_percentage: 0,
    soft_deleted_videos: 0
  },
  storage: {
    videos_gb: 0,
    generated_data_gb: 0,
    total_gb: 0
  },
  timestamp: new Date().toISOString(),
  region: 'loading...'
});

function formatTime(timestamp) {
  try {
    const date = new Date(timestamp);
    return date.toLocaleString();
  } catch {
    return timestamp;
  }
}

// TODO: Get actual user role from auth context/store
function checkAdminStatus() {
  // Placeholder: In real implementation, check user's account role
  // For now, assume non-admin, but admin users can see full dashboard
  isAdmin.value = localStorage.getItem('userRole') === 'admin' || false;
}

async function loadMonitoringStats() {
  try {
    const response = await getStats('monitoring');
    if (response && typeof response === 'object') {
      monitoring.value = {
        queue: response.queue || monitoring.value.queue,
        conflicts: response.conflicts || monitoring.value.conflicts,
        data: response.data || monitoring.value.data,
        storage: response.storage || monitoring.value.storage,
        timestamp: response.timestamp || new Date().toISOString(),
        region: response.region || 'unknown'
      };
    }
  } catch (e) {
    console.error('Failed to load monitoring stats:', e);
    // Keep previous values on error
  }
}

onMounted(() => {
  checkAdminStatus();
  loadMonitoringStats();
  // Refresh every 30 seconds
  setInterval(loadMonitoringStats, 30000);
});
</script>

<style scoped>
.monitoring-dashboard {
  padding: 1rem;
}

.card {
  background: white;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}
</style>

