<template>
  <div class="page-header">
    <div class="page-header-left">
      <div class="page-icon-wrap">
        <el-icon><component :is="icon" /></el-icon>
      </div>
      <div>
        <h1 class="page-title">{{ title }}</h1>
        <p class="page-desc">{{ desc }}</p>
      </div>
    </div>
    <slot />
  </div>
</template>

<script setup>
defineProps({
  title: { type: String, required: true },
  desc: { type: String, default: '' },
  icon: { type: String, default: 'Grid' },
  gradient: { type: String, default: 'blue' },   // blue | teal | full | lavender
})
</script>

<style scoped>
.page-icon-wrap {
  width: 44px; height: 44px; border-radius: var(--radius-lg);
  display: flex; align-items: center; justify-content: center;
  font-size: 22px;
  background: var(--primary-bg);
  color: var(--primary);
  border: 1px solid var(--primary-border);
}
</style>
