import { defineStore } from 'pinia'
import { ref } from 'vue'
import { ElNotification } from 'element-plus'
import { getJob, submitJob, cancelJob } from '../api/inference.js'

let _visHandler = null
let _registered = false

export const useJobStore = defineStore('job', () => {
  const jobs = ref({})       // jobId → job object
  const pollers = ref({})    // jobId → intervalId

  function _pauseAll() {
    Object.values(pollers.value).forEach(id => clearInterval(id))
  }
  function _resumeAll() {
    const ids = Object.keys(pollers.value)
    ids.forEach(jobId => {
      pollers.value[jobId] = _makeInterval(jobId)
    })
  }

  if (!_registered && typeof document !== 'undefined') {
    _visHandler = () => { document.hidden ? _pauseAll() : _resumeAll() }
    document.addEventListener('visibilitychange', _visHandler)
    _registered = true
  }

  // submit: error handling delegated to client.js interceptor
  async function submit(caseId, modelName) {
    const { data } = await submitJob(caseId, modelName)
    jobs.value[data.id] = data
    startPolling(data.id)
    return data
  }

  function _makeInterval(jobId) {
    return setInterval(async () => {
      if (document.hidden) return
      try {
        const { data } = await getJob(jobId)
        const prev = jobs.value[jobId]
        jobs.value[jobId] = data

        if (prev?.status === 'running' && data.status === 'done') {
          ElNotification({ title: '推理完成', message: `模型 ${data.model_name} 已完成分割，可查看结果`, type: 'success', duration: 5000 })
        } else if (prev?.status === 'running' && data.status === 'failed') {
          ElNotification({ title: '推理失败', message: data.error_message || '未知错误', type: 'error', duration: 8000 })
        }

        if (['done', 'failed', 'cancelled'].includes(data.status)) {
          stopPolling(jobId)
        }
      } catch {
        // 404 / network errors already handled by client interceptor
        stopPolling(jobId)
        delete jobs.value[jobId]
      }
    }, 1500)
  }

  function startPolling(jobId) {
    if (pollers.value[jobId]) return
    pollers.value[jobId] = _makeInterval(jobId)
  }

  function stopPolling(jobId) {
    if (pollers.value[jobId]) {
      clearInterval(pollers.value[jobId])
      delete pollers.value[jobId]
    }
  }

  async function cancel(jobId) {
    try {
      await cancelJob(jobId)
    } catch {
      // error handled by client interceptor
      return
    }
    stopPolling(jobId)
    if (jobs.value[jobId]) jobs.value[jobId].status = 'cancelled'
  }

  return { jobs, submit, startPolling, stopPolling, cancel }
})
