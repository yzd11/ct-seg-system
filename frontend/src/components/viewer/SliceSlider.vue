<template>
  <div class="slice-slider">
    <span class="label">切片</span>
    <el-slider
      :model-value="viewer.currentSlice"
      :min="0"
      :max="Math.max(viewer.totalSlices - 1, 0)"
      :step="1"
      show-input
      :input-size="'small'"
      style="flex: 1"
      @input="viewer.setSlice($event)"
      @change="viewer.setSlice($event)"
    />
    <span class="count">/ {{ viewer.totalSlices }}</span>
  </div>
</template>

<script setup>
import { useViewerStore } from '../../stores/viewerStore.js'
const viewer = useViewerStore()
</script>

<style scoped>
.slice-slider {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 0;
}
.label { color: #606266; font-size: 13px; white-space: nowrap; }
.count { color: #909399; font-size: 12px; white-space: nowrap; }
</style>
