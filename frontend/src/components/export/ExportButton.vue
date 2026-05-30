<template>
  <el-button
    :disabled="!jobId || !isDone"
    :loading="exporting"
    @click="doExport"
    type="success"
    style="width:100%"
  >
    <el-icon v-if="!exporting" style="margin-right:6px"><Download /></el-icon>
    {{ exporting ? '正在生成...' : '导出 PDF 报告' }}
  </el-button>
</template>

<script setup>
import { ref } from 'vue'
import client from '../../api/client.js'

const props = defineProps({
  jobId: { type: String, default: null },
  isDone: { type: Boolean, default: false },
})

const exporting = ref(false)

async function doExport() {
  if (!props.jobId) return
  exporting.value = true
  try {
    const res = await client.post(`/export/pdf?job_id=${props.jobId}`, null, { responseType: 'blob' })
    const url = URL.createObjectURL(new Blob([res.data], { type: 'application/pdf' }))
    const a = document.createElement('a')
    a.href = url
    a.download = `report_${props.jobId.slice(0, 8)}.pdf`
    a.click()
    URL.revokeObjectURL(url)
  } catch {
    // error handled by client.js interceptor
  } finally {
    exporting.value = false
  }
}
</script>
