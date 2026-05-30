<template>
  <div class="compare-view">

    <PageHeader title="多模型并排对比" desc="同一 CT 影像，并排对比不同模型的分割效果与量化指标" icon="Grid" />

    <!-- Controls -->
    <div class="controls-card glass-card">
      <div class="controls-inner">
        <div class="control-group">
          <label class="ctrl-label">选择案例</label>
          <el-select v-model="selectedCase" placeholder="请选择案例" style="width:200px" @change="resetJobs">
            <el-option v-for="c in caseStore.cases" :key="c.id" :label="c.patient_id" :value="c.id" />
          </el-select>
        </div>
        <div class="control-group">
          <label class="ctrl-label">选择模型（最多 3 个）</label>
          <el-select v-model="selectedModels" multiple placeholder="选择推理模型"
            style="width:340px" :multiple-limit="3" @change="resetJobs">
            <el-option v-for="m in MODEL_LIST" :key="m.value" :label="m.label" :value="m.value" />
          </el-select>
        </div>
        <el-button type="primary" size="large"
          :disabled="!selectedCase || !selectedModels.length"
          :loading="isRunning"
          @click="startAll">
          <el-icon style="margin-right:6px"><VideoPlay /></el-icon>
          {{ isRunning ? '推理中...' : '全部推理' }}
        </el-button>
      </div>
    </div>

    <!-- Slice slider (shared across all canvases) -->
    <div v-if="selectedCase" class="slider-card glass-card">
      <SliceSlider />
    </div>

    <!-- Canvas grid -->
    <div v-if="selectedModels.length" class="compare-grid" :class="`cols-${selectedModels.length}`">
      <div v-for="m in selectedModels" :key="m" class="compare-col glass-card">
        <div class="col-header">
          <div class="col-model-badge" :style="{ borderLeft: `3px solid ${MODEL_COLORS[m]}` }">
            {{ modelLabel(m) }}
          </div>
          <span v-if="jobs[m]" class="col-status" :class="'cst-' + jobs[m].status">
            {{ statusLabel(jobs[m].status) }}
          </span>
        </div>
        <div class="col-canvas">
          <SliceCanvas
            v-if="selectedCase"
            :case-id="selectedCase"
            :slice-index="viewer.currentSlice"
            :job-id="jobs[m]?.id || null"
          />
        </div>
        <div class="col-footer">
          <ProgressBar :job="jobs[m] || null" />
        </div>
      </div>
    </div>

    <div v-else class="compare-empty glass-card">
      <el-icon style="font-size:40px;color:#d1d5db"><Grid /></el-icon>
      <div class="empty-text">请选择案例和模型，然后点击「全部推理」</div>
    </div>

    <!-- ── 量化对比区（所有模型完成后显示） ──────────────────────────── -->
    <template v-if="allDone">

      <!-- 指标对比表 -->
      <div class="metrics-compare glass-card">
        <div class="mc-header">
          <div class="mc-header-left">
            <div class="mc-icon"><el-icon><DataAnalysis /></el-icon></div>
            <span class="mc-title">量化指标对比</span>
          </div>
          <span class="mc-badge">Quantitative Comparison</span>
        </div>

        <div class="mc-body">
          <table class="compare-table">
            <thead>
              <tr>
                <th class="metric-col">指标</th>
                <th v-for="m in selectedModels" :key="m" class="model-col">
                  <span class="model-dot" :style="{ background: MODEL_COLORS[m] }" />
                  {{ modelLabel(m) }}
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in metricRows" :key="row.key" :class="row.highlight ? 'row-highlight' : ''">
                <td class="metric-name">
                  <span>{{ row.label }}</span>
                  <span class="metric-unit">{{ row.unit }}</span>
                </td>
                <td v-for="m in selectedModels" :key="m" class="metric-val"
                  :class="isBest(row, m) ? 'val-best' : ''">
                  {{ row.fmt(jobs[m], sliceResults[m]) }}
                  <span v-if="isBest(row, m)" class="best-badge">最优</span>
                </td>
              </tr>
            </tbody>
          </table>
          <div class="table-note">
            <el-icon style="color:#94afc8;margin-right:5px"><InfoFilled /></el-icon>
            临床测量值（体积、FLR 等）均由各模型分割结果推算，无 ground truth 对比，仅供横向参考，不反映模型准确度。
            <b style="color:#2563eb">推理耗时</b>为系统性能指标，越低越优。
          </div>
        </div>
      </div>

      <!-- 肿瘤截面面积曲线对比 -->
      <div v-if="hasSliceData" class="area-compare glass-card">
        <div class="mc-header">
          <div class="mc-header-left">
            <div class="mc-icon"><el-icon><TrendCharts /></el-icon></div>
            <span class="mc-title">各模型肿瘤截面积曲线对比</span>
          </div>
          <span class="mc-badge">Area Curve</span>
        </div>
        <div class="mc-body chart-body">
          <Line :data="areaChartData" :options="areaChartOptions" />
        </div>
      </div>

    </template>

  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, onMounted, onUnmounted } from 'vue'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS, LineElement, PointElement, LinearScale, CategoryScale,
  Title, Tooltip, Legend, Filler,
} from 'chart.js'
import { useCaseStore } from '../stores/caseStore.js'
import { useViewerStore } from '../stores/viewerStore.js'
import { useJobStore } from '../stores/jobStore.js'
import { MODEL_LIST, MODEL_COLORS, MODEL_MAP } from '../constants/models.js'
import PageHeader from '../components/common/PageHeader.vue'
import SliceCanvas from '../components/viewer/SliceCanvas.vue'
import SliceSlider from '../components/viewer/SliceSlider.vue'
import ProgressBar from '../components/inference/ProgressBar.vue'
import client from '../api/client.js'

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Title, Tooltip, Legend, Filler)

const caseStore = useCaseStore()
const viewer    = useViewerStore()
const jobStore  = useJobStore()

const selectedCase   = ref(null)
const selectedModels = ref([])
const jobs           = reactive({})      // modelName → job object
const sliceResults   = ref({})           // modelName → slice result array

const modelLabel  = v => MODEL_MAP[v] || v
const statusLabel = s => ({ queued:'排队', running:'推理中', done:'完成', failed:'失败', cancelled:'已取消' }[s] ?? s)

const isRunning = computed(() =>
  selectedModels.value.some(m => ['queued','running'].includes(jobs[m]?.status))
)
const allDone = computed(() =>
  selectedModels.value.length > 0 &&
  selectedModels.value.every(m => jobs[m]?.status === 'done')
)
const hasSliceData = computed(() =>
  selectedModels.value.some(m => sliceResults.value[m]?.length > 0)
)

// 所有模型完成后批量拉取切片结果
watch(allDone, async (val) => {
  if (!val) return
  const fetches = selectedModels.value.map(async m => {
    if (!jobs[m]?.id) return
    try {
      const { data } = await client.get(`/inference/jobs/${jobs[m].id}/results`)
      sliceResults.value = { ...sliceResults.value, [m]: data }
    } catch { /* ignore */ }
  })
  await Promise.all(fetches)
})

// ── 量化指标定义 ─────────────────────────────────────────────────────────────
function _inferSecs(job) {
  if (!job?.started_at || !job?.finished_at) return null
  return ((new Date(job.finished_at) - new Date(job.started_at)) / 1000).toFixed(1)
}
function _tumorCount(sr) { return sr?.filter(r => r.tumor_area_px > 0).length ?? 0 }
function _maxTumor(sr)   { return sr ? Math.max(0, ...sr.map(r => r.tumor_area_px)) : 0 }
function _flr(job) {
  if (!job?.liver_volume_ml) return null
  return ((job.liver_volume_ml - (job.tumor_volume_ml || 0)) / job.liver_volume_ml * 100)
}
function _burden(job) {
  if (!job?.liver_volume_ml) return null
  return ((job.tumor_volume_ml || 0) / job.liver_volume_ml * 100)
}

// better: 'min'/'max' 表示有明确优劣方向（需客观依据）
// better: null 表示纯测量值，无 ground truth 无法判断优劣
const metricRows = [
  {
    key: 'liver', label: '肝脏体积', unit: 'mL', highlight: false, better: null,
    fmt: (j) => j?.liver_volume_ml != null ? j.liver_volume_ml.toFixed(1) : '—',
    val: (j) => j?.liver_volume_ml ?? null,
  },
  {
    key: 'tumor', label: '肿瘤体积', unit: 'mL', highlight: false, better: null,
    fmt: (j) => j?.tumor_volume_ml != null ? j.tumor_volume_ml.toFixed(1) : '—',
    val: (j) => j?.tumor_volume_ml ?? null,
  },
  {
    key: 'burden', label: '肿瘤负荷', unit: '%', highlight: false, better: null,
    fmt: (j) => { const v = _burden(j); return v != null ? v.toFixed(2) : '—' },
    val: (j) => _burden(j),
  },
  {
    key: 'flr', label: '剩余肝体积 FLR', unit: '%', highlight: true, better: null,
    fmt: (j) => { const v = _flr(j); return v != null ? v.toFixed(1) : '—' },
    val: (j) => _flr(j),
  },
  {
    key: 'slices', label: '含肿瘤切片数', unit: '张', highlight: false, better: null,
    fmt: (_, sr) => sr ? `${_tumorCount(sr)} / ${sr.length}` : '—',
    val: (_, sr) => _tumorCount(sr),
  },
  {
    key: 'maxarea', label: '最大截面积', unit: 'px²', highlight: false, better: null,
    fmt: (_, sr) => sr ? _maxTumor(sr).toLocaleString() : '—',
    val: (_, sr) => _maxTumor(sr),
  },
  {
    key: 'time', label: '推理耗时', unit: 's', highlight: false, better: 'min',
    fmt: (j) => { const v = _inferSecs(j); return v != null ? v : '—' },
    val: (j) => _inferSecs(j) != null ? Number(_inferSecs(j)) : null,
  },
]

function isBest(row, m) {
  if (!row.better) return false
  const vals = selectedModels.value
    .map(x => row.val(jobs[x], sliceResults.value[x]))
    .filter(v => v != null)
  if (vals.length < 2) return false
  const myVal = row.val(jobs[m], sliceResults.value[m])
  if (myVal == null) return false
  return row.better === 'min'
    ? myVal === Math.min(...vals)
    : myVal === Math.max(...vals)
}

// ── 面积曲线图 ────────────────────────────────────────────────────────────────
const areaChartData = computed(() => {
  const modelsWith = selectedModels.value.filter(m => sliceResults.value[m]?.length)
  if (!modelsWith.length) return { labels: [], datasets: [] }

  // 用最长的那个模型的切片索引作为 X 轴
  const longest = modelsWith.reduce((a, b) =>
    (sliceResults.value[a]?.length ?? 0) >= (sliceResults.value[b]?.length ?? 0) ? a : b
  )
  const step   = Math.max(1, Math.floor((sliceResults.value[longest]?.length ?? 1) / 80))
  const labels = (sliceResults.value[longest] ?? [])
    .filter((_, i) => i % step === 0).map(r => r.slice_index)

  const datasets = modelsWith.map(m => {
    const sr = sliceResults.value[m] ?? []
    const sampled = sr.filter((_, i) => i % step === 0)
    return {
      label: modelLabel(m),
      data: sampled.map(r => r.tumor_area_px),
      borderColor: MODEL_COLORS[m],
      backgroundColor: MODEL_COLORS[m] + '15',
      fill: true, tension: 0.4, pointRadius: 0, borderWidth: 1.8,
    }
  })
  return { labels, datasets }
})

const areaChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  plugins: {
    legend: {
      position: 'top',
      labels: { font: { size: 11, family: 'Inter' }, color: '#374151', boxWidth: 12, padding: 14 },
    },
    title: {
      display: true,
      text: '各模型肿瘤截面积对比曲线（px²）',
      font: { size: 12, family: 'Inter', weight: '700' },
      color: '#111827', padding: { bottom: 8 },
    },
    tooltip: {
      mode: 'index', intersect: false,
      callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toLocaleString()} px²` },
    },
  },
  scales: {
    x: { ticks: { color: '#9ca3af', font: { size: 10 }, maxTicksLimit: 8 }, grid: { color: 'rgba(0,0,0,0.04)' } },
    y: { ticks: { color: '#9ca3af', font: { size: 10 }, maxTicksLimit: 5 }, grid: { color: 'rgba(0,0,0,0.04)' } },
  },
}

// ── 任务管理 ──────────────────────────────────────────────────────────────────
function resetJobs() {
  selectedModels.value.forEach(m => delete jobs[m])
  sliceResults.value = {}
}

let _syncInterval = null
onMounted(() => caseStore.fetchCases())
onUnmounted(() => {
  if (_syncInterval) clearInterval(_syncInterval)
  // Stop jobStore polling for all jobs we submitted
  selectedModels.value.forEach(m => {
    if (jobs[m]?.id) jobStore.stopPolling(jobs[m].id)
  })
})

// 选择案例后加载切片数，驱动 SliceSlider
watch(selectedCase, async (id) => {
  viewer.resetState()
  if (!id) return
  try {
    const { data } = await client.get(`/nifti/${id}/metadata`)
    viewer.totalSlices = data.slice_count || 0
  } catch { /* ignore */ }
})

async function startAll() {
  if (_syncInterval) { clearInterval(_syncInterval); _syncInterval = null }
  sliceResults.value = {}

  for (const m of selectedModels.value) {
    try {
      const job = await jobStore.submit(selectedCase.value, m)
      jobs[m] = job
    } catch { /* jobStore already shows error toast */ }
  }

  _syncInterval = setInterval(() => {
    if (document.hidden) return
    for (const m of selectedModels.value) {
      if (jobs[m]?.id && jobStore.jobs[jobs[m].id]) {
        jobs[m] = jobStore.jobs[jobs[m].id]
      }
    }
    if (selectedModels.value.every(m => !jobs[m] || ['done','failed','cancelled'].includes(jobs[m].status))) {
      clearInterval(_syncInterval)
      _syncInterval = null
    }
  }, 1000)
}
</script>

<style scoped>
.compare-view { animation: page-fade-enter-active 0.3s ease both; display: flex; flex-direction: column; gap: 14px; }

/* Controls */
.controls-card { padding: 18px 24px; }
.controls-inner { display: flex; align-items: flex-end; gap: 20px; flex-wrap: wrap; }
.control-group  { display: flex; flex-direction: column; gap: 6px; }
.ctrl-label     { font-size: 12px; font-weight: 600; color: var(--text-secondary); }

.slider-card { padding: 14px 18px; }

/* Canvas grid */
.compare-grid { display: grid; gap: 14px; }
.compare-grid.cols-1 { grid-template-columns: 1fr; max-width: 520px; margin: 0 auto; }
.compare-grid.cols-2 { grid-template-columns: repeat(2, 1fr); }
.compare-grid.cols-3 { grid-template-columns: repeat(3, 1fr); }

.compare-col { padding: 0; overflow: hidden; }
.col-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 11px 14px; border-bottom: 1px solid var(--border-light);
  background: var(--gray-50);
}
.col-model-badge {
  font-family: var(--font-ui); font-size: 14px; font-weight: 700;
  color: var(--text-primary); padding-left: 10px;
}
.col-status {
  font-size: 11.5px; font-weight: 600; padding: 2px 9px;
  border-radius: 99px; font-family: var(--font-data);
  border: 1px solid transparent;
}
.cst-running  { background: var(--warning-bg); color: var(--warning); border-color: rgba(217, 119, 6, 0.2); }
.cst-done     { background: var(--success-bg); color: var(--success); border-color: rgba(5, 150, 105, 0.2); }
.cst-failed   { background: var(--danger-bg);  color: var(--danger);  border-color: rgba(220, 38, 38, 0.2); }
.cst-queued   { background: var(--primary-bg); color: var(--primary); border-color: var(--primary-border); }
.cst-cancelled{ background: var(--gray-100);   color: var(--text-secondary); border-color: var(--border-light); }

.col-canvas  { aspect-ratio: 1 / 1; min-height: 320px; }
.col-footer  { padding: 12px 14px; border-top: 1px solid var(--border-light); background: var(--gray-50); }

.compare-empty {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; padding: 60px; gap: 14px;
}
.empty-text { font-size: 14px; color: var(--text-disabled); }

/* ── responsive ────────────────────────────────────────── */
@media (max-width: 1280px) {
  .compare-grid.cols-3 { grid-template-columns: repeat(2, 1fr); }
  .compare-grid.cols-2 { grid-template-columns: 1fr; max-width: 520px; margin: 0 auto; }
}
@media (max-width: 768px) {
  .compare-grid.cols-2,
  .compare-grid.cols-3 { grid-template-columns: 1fr; max-width: 100%; }
  .col-canvas { aspect-ratio: 1/1; min-height: 240px; }
  .controls-inner .el-select { width: 100% !important; }
}

/* ── Metrics compare ────────────────────────────────────────── */
.metrics-compare, .area-compare { overflow: hidden; }

.mc-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 13px 18px; border-bottom: 1px solid var(--border-light);
  background: var(--gray-50);
}
.mc-header-left { display: flex; align-items: center; gap: 10px; }
.mc-icon {
  width: 28px; height: 28px; border-radius: var(--radius-sm);
  background: var(--primary-bg); color: var(--primary);
  display: flex; align-items: center; justify-content: center; font-size: 14px;
}
.mc-title {
  font-family: var(--font-ui); font-size: 15px;
  font-weight: 700; color: var(--text-primary);
}
.mc-badge {
  padding: 2px 9px; border-radius: 99px; font-size: 10px; font-weight: 600;
  font-family: var(--font-data);
  background: var(--gray-100); color: var(--text-tertiary);
  border: 1px solid var(--border-light);
}
.mc-body { padding: 16px 18px; }

/* Comparison table */
.compare-table {
  width: 100%; border-collapse: collapse; font-size: 13px;
}
.compare-table thead tr {
  background: var(--gray-50);
}
.compare-table th {
  padding: 10px 14px; text-align: left; font-weight: 700;
  color: var(--text-primary); font-family: var(--font-ui); font-size: 13px;
  border-bottom: 2px solid var(--border-light);
}
.metric-col { width: 160px; }
.model-col  { min-width: 130px; }

.compare-table td {
  padding: 9px 14px; border-bottom: 1px solid var(--gray-100);
  vertical-align: middle;
}
.row-highlight td { background: var(--gray-50); }
.metric-name { display: flex; align-items: center; gap: 6px; color: var(--text-secondary); font-weight: 500; }
.metric-unit { font-size: 11px; color: var(--text-tertiary); font-family: var(--font-data); }
.metric-val  {
  font-family: var(--font-data); font-weight: 600;
  color: var(--text-primary); position: relative;
}
.val-best { color: var(--success) !important; }
.best-badge {
  display: inline-block; margin-left: 6px;
  background: var(--success-bg); color: var(--success);
  border: 1px solid rgba(5, 150, 105, 0.2);
  border-radius: 4px; padding: 1px 5px;
  font-size: 10px; font-weight: 700;
  font-family: var(--font-ui);
}
.model-dot {
  display: inline-block; width: 8px; height: 8px;
  border-radius: 50%; margin-right: 6px; vertical-align: middle;
}

/* Table note */
.table-note {
  display: flex; align-items: flex-start; gap: 4px;
  margin-top: 12px; padding: 10px 14px;
  background: var(--gray-50);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-md);
  font-size: 12px; color: var(--text-tertiary); line-height: 1.6;
}

/* Area chart */
.chart-body { height: 240px; }
</style>
