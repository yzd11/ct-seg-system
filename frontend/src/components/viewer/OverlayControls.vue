<template>
  <div class="overlay-controls">
    <span class="label">叠加层</span>
    <el-checkbox v-model="viewer.showLiver">
      <span style="color: #10b981">■</span> 肝脏
    </el-checkbox>
    <el-checkbox v-model="viewer.showTumor">
      <span style="color: #ef4444">■</span> 肿瘤
    </el-checkbox>
    <span class="label" style="margin-left: 12px">透明度</span>
    <el-slider v-model="viewer.overlayOpacity" :min="0" :max="1" :step="0.05" style="width: 120px" />
  </div>
</template>

<script setup>
import { useViewerStore } from '../../stores/viewerStore.js'
const viewer = useViewerStore()
</script>

<style scoped>
.overlay-controls {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.label { color: #606266; font-size: 13px; }
</style>
