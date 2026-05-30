<template>
  <!-- Healing background layer -->
  <div class="app-bg" aria-hidden="true">
    <div class="mesh-grid" />
  </div>

  <!-- Layout -->
  <el-container style="height: 100vh; position: relative; z-index: 1">
    <el-header style="height: 60px; padding: 0; flex-shrink: 0">
      <AppHeader />
    </el-header>
    <el-container style="overflow: hidden">
      <el-aside width="216px">
        <AppSidebar />
      </el-aside>
      <el-main class="app-main">
        <router-view v-slot="{ Component }">
          <transition name="page-fade" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import AppHeader from './components/layout/AppHeader.vue'
import AppSidebar from './components/layout/AppSidebar.vue'
</script>

<style>
/* ── Clean diagnostic background ──────────────────────────── */
.app-bg {
  position: fixed;
  inset: 0;
  background: #f8fafc;
  z-index: 0;
  overflow: hidden;
}

/* Precision grid */
.mesh-grid {
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(0, 0, 0, 0.018) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 0, 0, 0.018) 1px, transparent 1px);
  background-size: 64px 64px;
}

/* ── Main content ──────────────────────────────────────────── */
.app-main {
  padding: 24px;
  overflow-y: auto;
  background: transparent !important;
}

/* el containers transparent */
.el-header    { background: transparent !important; }
.el-aside     { background: transparent !important; border-right: none !important; }
.el-main      { background: transparent !important; }
.el-container { background: transparent !important; }
</style>
