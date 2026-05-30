import axios from 'axios'
import { ElMessage, ElNotification } from 'element-plus'

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE ?? '/api/v1',
  timeout: 120_000,
})

// ── Request interceptor ──────────────────────────────────────
let _reqSeq = 0
client.interceptors.request.use((config) => {
  config.headers['X-Request-ID'] = `${Date.now()}-${++_reqSeq}`
  return config
})

// ── Response interceptor: unified 4xx/5xx/net error handling ─
client.interceptors.response.use(
  (res) => res,
  (error) => {
    if (error.response) {
      const { status, data, config } = error.response
      const detail = data?.detail ?? `请求失败 (${status})`

      if (status >= 500) {
        ElNotification({ title: '服务器异常', message: detail, type: 'error', duration: 6000 })
        // Retry once for 5xx
        return client.request(config)
      }
      if (status === 404) {
        ElMessage.warning(detail)
      } else if (status >= 400) {
        ElMessage.error(detail)
      }
    } else if (error.code === 'ECONNABORTED') {
      ElMessage.warning('请求超时，请检查网络后重试')
    } else {
      ElMessage.warning('网络异常，请检查连接')
    }
    return Promise.reject(error)
  },
)

export default client
