<template>
  <div v-if="job" class="progress-wrap">
    <div class="progress-header">
      <div class="progress-left">
        <span class="model-chip">{{ friendlyName }}</span>
        <span class="status-badge" :class="'s-' + job.status">
          <span v-if="job.status === 'running'" class="running-dot" />
          {{ statusLabel }}
        </span>
      </div>
      <span v-if="job.status === 'running'" class="slice-info">
        {{ job.current_slice }} / {{ job.total_slices }} 切片
      </span>
    </div>
    <div class="progress-bar-wrap">
      <el-progress
        :percentage="clampedProgress"
        :status="progressStatus"
        :color="job.status === 'cancelled' ? '#94a3b8' : undefined"
        :stroke-width="7"
        :show-text="false"
      />
      <span class="progress-pct">{{ clampedProgress }}%</span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { MODEL_MAP } from '../../constants/models.js'

const props = defineProps({ job: { type: Object, default: null } })

const friendlyName = computed(() =>
  MODEL_MAP[props.job?.model_name] || props.job?.model_name || ''
)

const statusLabel = computed(() => ({
  queued: '排队中', running: '推理中', done: '已完成', failed: '失败', cancelled: '已取消',
}[props.job?.status] ?? props.job?.status))

const clampedProgress = computed(() =>
  Math.min(100, Math.max(0, props.job?.progress ?? 0))
)

const progressStatus = computed(() => {
  if (props.job?.status === 'done')      return 'success'
  if (props.job?.status === 'failed')    return 'exception'
  if (props.job?.status === 'cancelled') return 'warning'
  return ''
})
</script>

<style scoped>
.progress-wrap { display: flex; flex-direction: column; gap: 8px; }

.progress-header {
  display: flex; justify-content: space-between; align-items: center;
  gap: 8px;
}
.progress-left { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }

.model-chip {
  background: var(--primary-bg);
  color: var(--primary);
  border: 1px solid var(--primary-border);
  border-radius: var(--radius-sm);
  padding: 2px 9px;
  font-size: 12px; font-weight: 600;
  font-family: var(--font-data);
}

.status-badge {
  display: flex; align-items: center; gap: 5px;
  padding: 2px 9px;
  border-radius: 99px;
  font-size: 12px; font-weight: 600;
  border: 1px solid transparent;
}
.s-running  { background: var(--warning-bg); color: var(--warning); border-color: rgba(217, 119, 6, 0.2); }
.s-done     { background: var(--success-bg); color: var(--success); border-color: rgba(5, 150, 105, 0.2); }
.s-failed   { background: var(--danger-bg);  color: var(--danger);  border-color: rgba(220, 38, 38, 0.2); }
.s-queued   { background: var(--primary-bg); color: var(--primary); border-color: var(--primary-border); }
.s-cancelled{ background: var(--gray-100);   color: var(--text-secondary); border-color: var(--border-light); }

.running-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--warning);
}

.slice-info {
  font-size: 12px; color: var(--text-tertiary);
  font-family: var(--font-data);
}

.progress-bar-wrap {
  display: flex; align-items: center; gap: 10px;
}
.progress-bar-wrap .el-progress { flex: 1; }
.progress-pct {
  font-size: 12px; font-weight: 700;
  color: var(--primary);
  font-family: var(--font-data);
  min-width: 36px; text-align: right;
}
</style>
