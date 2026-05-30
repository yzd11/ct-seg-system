<template>
  <div class="sidebar">

    <!-- Diagnosis tools -->
    <div class="section-label">诊断工具</div>
    <nav class="nav-list">
      <RouterLink
        v-for="item in diagItems" :key="item.to" :to="item.to"
        class="nav-item" :class="{ active: $route.path === item.to }"
      >
        <div class="nav-icon">
          <el-icon><component :is="item.icon" /></el-icon>
        </div>
        <span class="nav-label">{{ item.label }}</span>
      </RouterLink>
    </nav>

    <div class="section-divider" />

    <!-- Analysis tools -->
    <div class="section-label">数据分析</div>
    <nav class="nav-list">
      <RouterLink
        v-for="item in analysisItems" :key="item.to" :to="item.to"
        class="nav-item" :class="{ active: $route.path === item.to }"
      >
        <div class="nav-icon">
          <el-icon><component :is="item.icon" /></el-icon>
        </div>
        <span class="nav-label">{{ item.label }}</span>
      </RouterLink>
    </nav>

    <!-- Footer -->
    <div class="sidebar-footer">
      <div class="footer-card">
        <div class="footer-row">
          <span class="footer-key">版本</span>
          <span class="footer-val">v 1.0.0</span>
        </div>
        <div class="footer-row">
          <span class="footer-key">数据集</span>
          <span class="footer-val">LiTS-2017</span>
        </div>
        <div class="footer-bar-row">
          <span class="footer-key">在线状态</span>
          <div class="footer-bar">
            <div class="footer-bar-fill" style="width: 100%" />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
const diagItems = [
  { to: '/',       label: '案例列表', icon: 'House'  },
  { to: '/upload', label: '上传 CT',  icon: 'Upload' },
]
const analysisItems = [
  { to: '/compare', label: '多模型对比', icon: 'Grid'  },
  { to: '/history', label: '历史记录',   icon: 'Clock' },
]
</script>

<style scoped>
.sidebar {
  height: 100%;
  background: var(--bg-card);
  border-right: 1px solid var(--border-light);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding-bottom: var(--space-md);
}

/* Section label */
.section-label {
  font-family: var(--font-ui);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 1px;
  color: var(--text-tertiary);
  padding: 16px 18px 6px;
}

.section-divider {
  height: 1px;
  background: var(--border-light);
  margin: 6px 14px 4px;
}

/* Nav */
.nav-list { display: flex; flex-direction: column; gap: 2px; padding: 0 10px; }

.nav-item {
  display: flex;
  align-items: center;
  gap: 11px;
  padding: 10px 12px;
  border-radius: var(--radius-md);
  text-decoration: none;
  color: var(--text-secondary);
  font-size: 14px;
  font-weight: 500;
  transition: all var(--ease);
  border: 1px solid transparent;
}
.nav-item:hover {
  background: var(--gray-50);
  color: var(--text-primary);
}
.nav-item.active {
  background: var(--primary-bg);
  color: var(--primary);
  border-color: var(--primary-border);
  font-weight: 600;
}

/* Icon circles — unified style */
.nav-icon {
  width: 34px; height: 34px;
  border-radius: var(--radius-md);
  display: flex; align-items: center; justify-content: center;
  font-size: 16px;
  flex-shrink: 0;
  transition: all var(--ease);
  background: var(--gray-100);
  color: var(--text-tertiary);
}
.nav-item:hover .nav-icon {
  background: var(--gray-200);
  color: var(--text-secondary);
}
.nav-item.active .nav-icon {
  background: var(--primary);
  color: #fff;
}

.nav-label { flex: 1; }

/* Footer */
.sidebar-footer { margin-top: auto; padding: 0 12px; }
.footer-card {
  background: var(--gray-50);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-md);
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.footer-row { display: flex; justify-content: space-between; align-items: center; }
.footer-key { font-size: 11px; color: var(--text-tertiary); }
.footer-val { font-size: 11px; font-family: var(--font-data); color: var(--text-secondary); font-weight: 500; }
.footer-bar-row { display: flex; flex-direction: column; gap: 4px; }
.footer-bar {
  height: 3px;
  background: var(--gray-200);
  border-radius: 99px;
  overflow: hidden;
}
.footer-bar-fill {
  height: 100%;
  background: var(--success);
  border-radius: 99px;
}
</style>
