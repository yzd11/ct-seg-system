<template>
  <div class="model-selector">
    <el-select v-model="selected" placeholder="选择模型" style="width: 200px">
      <el-option v-for="m in models" :key="m.value" :label="m.label" :value="m.value" />
    </el-select>
    <el-button type="primary" :loading="running" @click="submit">开始推理</el-button>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useJobStore } from '../../stores/jobStore.js'
import { useViewerStore } from '../../stores/viewerStore.js'
import { ElMessage } from 'element-plus'
import { MODEL_LIST } from '../../constants/models.js'

const props = defineProps({ caseId: { type: String, required: true } })

const models = MODEL_LIST

const selected = ref('att_unet_pp')
const running = ref(false)
const jobStore = useJobStore()
const viewer = useViewerStore()

async function submit() {
  if (!selected.value) return
  running.value = true
  try {
    const job = await jobStore.submit(props.caseId, selected.value)
    viewer.activeJobId = job.id
    ElMessage.success('推理任务已提交')
  } catch {
    // error handled by client.js interceptor
  } finally {
    running.value = false
  }
}
</script>

<style scoped>
.model-selector { display: flex; gap: 8px; align-items: center; }
</style>
