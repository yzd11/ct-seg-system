<template>
  <div class="windowing">
    <span class="label">窗位</span>
    <el-input-number v-model="viewer.windowCenter" :min="-1000" :max="3071" :step="10" size="small" style="width: 110px" />
    <span class="label">窗宽</span>
    <el-input-number v-model="viewer.windowWidth" :min="1" :max="4000" :step="50" size="small" style="width: 110px" />
    <el-button size="small" @click="reset">重置</el-button>
  </div>
</template>

<script setup>
import { useViewerStore } from '../../stores/viewerStore.js'
const viewer = useViewerStore()
function reset() {
  viewer.windowCenter = 50
  viewer.windowWidth = 400
}
</script>

<style scoped>
.windowing {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.label { color: #606266; font-size: 13px; white-space: nowrap; }
</style>
