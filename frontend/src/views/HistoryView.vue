<template>
  <div class="history-view">

    <PageHeader title="历史推理记录" desc="查看所有 AI 分割推理任务的执行历史与结果" icon="Clock" />

    <!-- Filter bar -->
    <div class="filter-bar glass-card">
      <div class="filter-left">
        <el-icon style="color:#94afc8"><Filter /></el-icon>
        <span class="filter-label">筛选案例</span>
        <el-select
          v-model="selectedCase"
          placeholder="全部案例"
          clearable
          size="default"
          style="width:200px"
          @change="loadJobs"
        >
          <el-option v-for="c in caseStore.cases" :key="c.id" :label="c.patient_id" :value="c.id" />
        </el-select>
      </div>
      <div class="filter-right">
        <span class="record-count">共 {{ jobList.length }} 条记录</span>
      </div>
    </div>

    <!-- Table -->
    <div class="table-wrap glass-card">
      <el-table :data="pagedList" style="width:100%;background:transparent">
        <el-table-column label="案例 ID" min-width="120">
          <template #default="{ row }">
            <span class="mono-text">{{ row.case_id.slice(0, 8) }}…</span>
          </template>
        </el-table-column>

        <el-table-column prop="model_name" label="推理模型" width="160">
          <template #default="{ row }">
            <span class="model-chip">{{ row.model_name }}</span>
          </template>
        </el-table-column>

        <el-table-column label="状态" width="110">
          <template #default="{ row }">
            <div class="status-wrap">
              <span class="status-dot-sm" :class="'dot-' + row.status" />
              <el-tag :type="statusType(row.status)" size="small">{{ statusLabel(row.status) }}</el-tag>
            </div>
          </template>
        </el-table-column>

        <el-table-column label="肝脏体积" width="120" align="right">
          <template #default="{ row }">
            <span class="vol-text">{{ row.liver_volume_ml != null ? row.liver_volume_ml.toFixed(1) + ' mL' : '—' }}</span>
          </template>
        </el-table-column>

        <el-table-column label="肿瘤体积" width="120" align="right">
          <template #default="{ row }">
            <span class="vol-text vol-tumor">{{ row.tumor_volume_ml != null ? row.tumor_volume_ml.toFixed(1) + ' mL' : '—' }}</span>
          </template>
        </el-table-column>

        <el-table-column label="完成时间" min-width="180">
          <template #default="{ row }">
            <div class="time-cell">
              <span class="time-text">{{ fmtTime(row.finished_at) }}</span>
              <span v-if="inferDuration(row)" class="time-dur">{{ inferDuration(row) }}</span>
            </div>
          </template>
        </el-table-column>

        <el-table-column label="操作" width="150" fixed="right">
          <template #default="{ row }">
            <div class="action-btns">
              <el-button size="small" type="primary" plain @click="openViewer(row.case_id, row.id)">查看</el-button>
              <el-button size="small" type="danger" plain @click="deleteJob(row.id)">删除</el-button>
            </div>
          </template>
        </el-table-column>
      </el-table>
      <!-- Pagination -->
      <div class="pagination-wrap">
        <el-pagination
          v-model:current-page="currentPage"
          :page-size="PAGE_SIZE"
          :total="jobList.length"
          layout="total, prev, pager, next"
          background
        />
      </div>
    </div>

  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useCaseStore } from '../stores/caseStore.js'
import { useViewerStore } from '../stores/viewerStore.js'
import { cancelJob } from '../api/inference.js'
import { utcParse, relativeTime } from '../utils/time.js'
import PageHeader from '../components/common/PageHeader.vue'
import client from '../api/client.js'

const caseStore = useCaseStore()
const viewer = useViewerStore()
const router = useRouter()

const selectedCase = ref(null)
const jobList = ref([])
const currentPage = ref(1)
const PAGE_SIZE = 10

const pagedList = computed(() => {
  const start = (currentPage.value - 1) * PAGE_SIZE
  return jobList.value.slice(start, start + PAGE_SIZE)
})

onMounted(async () => {
  await caseStore.fetchCases()
  await loadJobs()
})

async function loadJobs() {
  currentPage.value = 1
  try {
    if (selectedCase.value) {
      const { data } = await client.get(`/inference/cases/${selectedCase.value}/jobs`)
      jobList.value = data
    } else {
      const results = await Promise.allSettled(
        caseStore.cases.map(c => client.get(`/inference/cases/${c.id}/jobs`))
      )
      const all = []
      results.forEach(r => { if (r.status === 'fulfilled') all.push(...r.value.data) })
      jobList.value = all.sort((a, b) => new Date(b.finished_at ?? b.created_at) - new Date(a.finished_at ?? a.created_at))
    }
  } catch {
    // error handled by client.js interceptor
  }
}

function fmtTime(s) {
  const d = utcParse(s)
  return d ? d.toLocaleString('zh-CN', {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
  }) : '—'
}

function inferDuration(row) {
  const start = utcParse(row.started_at)
  const end   = utcParse(row.finished_at)
  if (!start || !end) return null
  return `${((end - start) / 1000).toFixed(1)}s`
}

function statusType(s) {
  return { done: 'success', running: 'warning', failed: 'danger', cancelled: 'info', queued: '' }[s] ?? ''
}
function statusLabel(s) {
  return { done: '已完成', running: '推理中', failed: '失败', cancelled: '已取消', queued: '排队中' }[s] ?? s
}

function openViewer(caseId, jobId) {
  viewer.activeJobId = jobId
  router.push(`/viewer/${caseId}`)
}

async function deleteJob(jobId) {
  await ElMessageBox.confirm('确认删除该推理记录及所有分割结果？', '删除确认', { type: 'warning' })
  try {
    await cancelJob(jobId)
    jobList.value = jobList.value.filter(j => j.id !== jobId)
    ElMessage.success('已删除')
  } catch {
    // error handled by client.js interceptor
  }
}
</script>

<style scoped>
.history-view { animation: page-fade-enter-active 0.3s ease both; }

/* Filter bar */
.filter-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 20px;
  margin-bottom: 14px;
}
.filter-left { display: flex; align-items: center; gap: 10px; }
.filter-label { font-size: 13px; font-weight: 600; color: var(--text-secondary); }
.record-count {
  font-size: 12px;
  color: var(--text-tertiary);
  background: var(--gray-100);
  border: 1px solid var(--border-light);
  border-radius: 99px;
  padding: 3px 12px;
  font-family: var(--font-data);
}

/* Table wrapper */
.table-wrap { padding: 0; overflow: hidden; }

/* Table cell styles */
.mono-text {
  font-family: var(--font-data);
  font-size: 12.5px;
  color: var(--text-secondary);
}
.model-chip {
  background: var(--primary-bg);
  color: var(--primary);
  border: 1px solid var(--primary-border);
  border-radius: var(--radius-sm);
  padding: 3px 9px;
  font-size: 12px;
  font-weight: 600;
  font-family: var(--font-data);
}
.status-wrap { display: flex; align-items: center; gap: 6px; }
.status-dot-sm {
  width: 6px; height: 6px;
  border-radius: 50%; flex-shrink: 0;
}
.dot-done     { background: var(--success); }
.dot-running  { background: var(--warning); }
.dot-failed   { background: var(--danger); }
.dot-cancelled{ background: var(--gray-400); }
.dot-queued   { background: var(--primary-light); }

.vol-text {
  font-family: var(--font-data);
  font-size: 13px;
  font-weight: 600;
  color: var(--success);
}
.vol-tumor { color: var(--danger); }

.time-cell { display: flex; flex-direction: column; gap: 2px; }
.time-text {
  font-size: 12.5px;
  color: var(--text-tertiary);
  font-family: var(--font-data);
}
.time-dur {
  font-size: 11px;
  color: var(--text-tertiary);
  font-family: var(--font-data);
}

.action-btns { display: flex; gap: 6px; }

.pagination-wrap {
  display: flex;
  justify-content: flex-end;
  padding: 14px 16px;
  border-top: 1px solid var(--border-light);
  background: var(--gray-50);
}
</style>
