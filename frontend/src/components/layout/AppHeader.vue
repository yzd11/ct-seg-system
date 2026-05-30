<template>
  <div class="header">

    <!-- Left: brand -->
    <div class="header-left">
      <div class="brand-icon-wrap">
        <el-icon class="brand-icon"><FirstAidKit /></el-icon>
      </div>
      <div class="brand-text">
        <span class="brand-title">CT 肝脏与肿瘤分割系统</span>
      </div>
      <div class="header-badge">
        <span class="live-dot" />
        LiTS 2017
      </div>
    </div>

    <!-- Right -->
    <div class="header-right">
      <div class="status-chip">
        <span class="status-dot" />
        系统就绪
      </div>

      <div class="vline" />

      <div class="header-time">
        <el-icon style="margin-right:5px;color:#94afc8;font-size:13px"><Clock /></el-icon>
        {{ currentTime }}
      </div>

      <div class="vline" />

      <div class="header-user">
        <div class="user-avatar">管</div>
        <div class="user-info">
          <span class="user-name">管理员</span>
          <span class="user-role">Administrator</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const currentTime = ref('')
function updateTime() {
  currentTime.value = new Date().toLocaleString('zh-CN', {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    hour12: false,
  })
}
let timer = null

function startClock() {
  if (timer) return
  updateTime()
  timer = setInterval(updateTime, 1000)
}
function stopClock() {
  clearInterval(timer)
  timer = null
}
function onVisibilityChange() {
  document.hidden ? stopClock() : startClock()
}

onMounted(() => {
  startClock()
  document.addEventListener('visibilitychange', onVisibilityChange)
})
onUnmounted(() => {
  stopClock()
  document.removeEventListener('visibilitychange', onVisibilityChange)
})
</script>

<style scoped>
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 60px;
  padding: 0 28px;
  background: rgba(255, 255, 255, 0.95);
  border-bottom: 1px solid var(--border-light);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

/* ── Left: brand ─────────────────────────────────────────── */
.header-left {
  display: flex;
  align-items: center;
  gap: 14px;
}

.brand-icon-wrap {
  width: 38px; height: 38px;
  border-radius: var(--radius-md);
  background: var(--primary);
  display: flex; align-items: center; justify-content: center;
}
.brand-icon { font-size: 20px; color: #fff; }

.brand-text { display: flex; flex-direction: column; gap: 1px; }
.brand-title {
  font-family: var(--font-ui);
  font-size: 16px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.2px;
}
.header-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  background: var(--primary-bg);
  border: 1px solid var(--primary-border);
  border-radius: 99px;
  font-size: 11px;
  font-weight: 600;
  color: var(--primary);
  font-family: var(--font-ui);
}

/* ── Right ────────────────────────────────────────────────── */
.header-right {
  display: flex;
  align-items: center;
  gap: 18px;
}

.status-chip {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 5px 14px;
  background: var(--success-bg);
  border: 1px solid rgba(5, 150, 105, 0.2);
  border-radius: 99px;
  font-size: 12px;
  font-weight: 500;
  color: var(--success);
}
.status-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--success);
}

.vline { width: 1px; height: 22px; background: var(--border-light); }

.header-time {
  display: flex;
  align-items: center;
  font-family: var(--font-data);
  font-size: 12.5px;
  color: var(--text-tertiary);
  font-variant-numeric: tabular-nums;
}

.header-user {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 5px 12px 5px 5px;
  border-radius: var(--radius-md);
  border: 1px solid transparent;
  cursor: pointer;
  transition: all var(--ease);
}
.header-user:hover {
  background: var(--gray-50);
  border-color: var(--border-light);
}
.user-avatar {
  width: 32px; height: 32px;
  border-radius: 50%;
  background: var(--primary);
  color: #fff;
  font-size: 13px;
  font-weight: 600;
  display: flex; align-items: center; justify-content: center;
}
.user-info { display: flex; flex-direction: column; gap: 1px; }
.user-name { font-size: 13px; font-weight: 600; color: var(--text-primary); line-height: 1.2; }
.user-role { font-size: 10px; color: var(--text-tertiary); font-family: var(--font-ui); }
</style>
