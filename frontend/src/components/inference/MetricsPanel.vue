<template>
  <div class="metrics-panel">
    <div v-if="hasData">
      <!-- Stat chips row -->
      <div class="stat-chips">
        <div class="stat-chip chip-teal">
          <div class="chip-label">含肿瘤切片</div>
          <div class="chip-val">{{ tumorSliceCount }}<span class="chip-unit"> 张</span></div>
          <div class="chip-sub">共 {{ results.length }} 张（{{ tumorPct }}%）</div>
        </div>
        <div class="stat-chip chip-red">
          <div class="chip-label">最大肿瘤截面</div>
          <div class="chip-val">{{ maxTumorArea }}<span class="chip-unit"> px²</span></div>
          <div class="chip-sub">第 {{ maxTumorSlice }} 切片</div>
        </div>
        <div class="stat-chip chip-blue">
          <div class="chip-label">平均肿瘤面积</div>
          <div class="chip-val">{{ meanTumorArea }}<span class="chip-unit"> px²</span></div>
          <div class="chip-sub">仅含肿瘤切片</div>
        </div>
        <div class="stat-chip chip-lavender">
          <div class="chip-label">肿瘤/肝脏比</div>
          <div class="chip-val">{{ tumorLiverRatio }}<span class="chip-unit">%</span></div>
          <div class="chip-sub">体积占比估算</div>
        </div>
      </div>

      <!-- Perimeter row (shown if backend provides data) -->
      <div v-if="hasPerimeter" class="perim-row">
        <div class="perim-item">
          <span class="perim-label">肝脏周长总和</span>
          <span class="perim-val">{{ totalLiverPerim.toLocaleString() }} px</span>
        </div>
        <div class="perim-divider" />
        <div class="perim-item">
          <span class="perim-label">肿瘤周长总和</span>
          <span class="perim-val">{{ totalTumorPerim.toLocaleString() }} px</span>
        </div>
        <div class="perim-divider" />
        <div class="perim-item">
          <span class="perim-label">形状圆度指数</span>
          <span class="perim-val">{{ tumorCircularity }}</span>
        </div>
      </div>

      <!-- Area chart (compact) -->
      <div class="chart-wrap">
        <Line :data="chartData" :options="chartOptions" />
      </div>
    </div>
    <div v-else class="no-data">
      <el-icon style="font-size:28px;color:#d1d5db"><DataAnalysis /></el-icon>
      <span>推理完成后显示各切片面积曲线及多维指标</span>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS, LineElement, PointElement, LinearScale, CategoryScale,
  Title, Tooltip, Legend, Filler,
} from 'chart.js'
import { getJobResults } from '../../api/inference.js'

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Title, Tooltip, Legend, Filler)

const props = defineProps({ job: { type: Object, default: null } })
const results = ref([])

watch(() => props.job?.status, async (status) => {
  if (status === 'done' && props.job?.id) {
    try {
      const { data } = await getJobResults(props.job.id)
      results.value = data
    } catch { /* silently ignore */ }
  } else if (status !== 'done') {
    results.value = []
  }
}, { immediate: true })

const hasData = computed(() => results.value.length > 0)

// ── Perimeter availability ──────────────────────────────────
const hasPerimeter = computed(() =>
  results.value.some(r => (r.liver_perimeter_px || 0) > 0 || (r.tumor_perimeter_px || 0) > 0)
)
const totalLiverPerim = computed(() =>
  results.value.reduce((s, r) => s + (r.liver_perimeter_px || 0), 0)
)
const totalTumorPerim = computed(() =>
  results.value.reduce((s, r) => s + (r.tumor_perimeter_px || 0), 0)
)
const tumorCircularity = computed(() => {
  // Circularity = 4π·Area / Perimeter² — averaged over tumor-bearing slices
  const slices = results.value.filter(r => r.tumor_area_px > 0 && (r.tumor_perimeter_px || 0) > 0)
  if (!slices.length) return '—'
  const avg = slices.reduce((s, r) => {
    const c = (4 * Math.PI * r.tumor_area_px) / (r.tumor_perimeter_px ** 2)
    return s + c
  }, 0) / slices.length
  return avg.toFixed(3)
})

// ── Derived statistics ──────────────────────────────────────
const tumorSliceCount = computed(() =>
  results.value.filter(r => r.tumor_area_px > 0).length
)
const tumorPct = computed(() => {
  if (!results.value.length) return 0
  return ((tumorSliceCount.value / results.value.length) * 100).toFixed(1)
})
const maxTumorEntry = computed(() =>
  results.value.reduce((best, r) => r.tumor_area_px > (best?.tumor_area_px ?? 0) ? r : best, null)
)
const maxTumorArea = computed(() => maxTumorEntry.value?.tumor_area_px ?? 0)
const maxTumorSlice = computed(() => maxTumorEntry.value?.slice_index ?? '—')
const meanTumorArea = computed(() => {
  const withTumor = results.value.filter(r => r.tumor_area_px > 0)
  if (!withTumor.length) return 0
  return Math.round(withTumor.reduce((s, r) => s + r.tumor_area_px, 0) / withTumor.length)
})
const tumorLiverRatio = computed(() => {
  const totalLiver = results.value.reduce((s, r) => s + r.liver_area_px, 0)
  const totalTumor = results.value.reduce((s, r) => s + r.tumor_area_px, 0)
  if (!totalLiver) return '—'
  return ((totalTumor / totalLiver) * 100).toFixed(2)
})

// ── Chart data ──────────────────────────────────────────────
const chartData = computed(() => {
  const step = Math.max(1, Math.floor(results.value.length / 80))
  const sampled = results.value.filter((_, i) => i % step === 0)
  return {
    labels: sampled.map(r => r.slice_index),
    datasets: [
      {
        label: '肝脏',
        data: sampled.map(r => r.liver_area_px),
        borderColor: '#0d9488',
        backgroundColor: 'rgba(13,148,136,0.07)',
        fill: true, tension: 0.4, pointRadius: 0, borderWidth: 1.5,
      },
      {
        label: '肿瘤',
        data: sampled.map(r => r.tumor_area_px),
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239,68,68,0.07)',
        fill: true, tension: 0.4, pointRadius: 0, borderWidth: 1.5,
      },
    ],
  }
})

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  plugins: {
    legend: {
      position: 'top',
      labels: { font: { size: 10, family: 'Inter' }, color: '#374151', boxWidth: 10, padding: 10 },
    },
    title: {
      display: true,
      text: '各切片分割面积曲线 (px²)',
      font: { size: 11, family: 'Inter', weight: '700' },
      color: '#111827',
      padding: { bottom: 6 },
    },
  },
  scales: {
    x: {
      title: { display: false },
      ticks: { color: '#9ca3af', font: { size: 9 }, maxTicksLimit: 6 },
      grid: { color: 'rgba(0,0,0,0.04)' },
    },
    y: {
      title: { display: false },
      ticks: { color: '#9ca3af', font: { size: 9 }, maxTicksLimit: 4 },
      grid: { color: 'rgba(0,0,0,0.04)' },
    },
  },
}
</script>

<style scoped>
.metrics-panel { display: flex; flex-direction: column; gap: 10px; }

/* Stat chips */
.stat-chips {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
}
.stat-chip {
  border-radius: var(--radius-md);
  padding: 9px 11px;
  border: 1px solid transparent;
  display: flex; flex-direction: column; gap: 2px;
}
.chip-teal   { background: var(--success-bg); border-color: rgba(5, 150, 105, 0.12); }
.chip-red    { background: var(--danger-bg);  border-color: rgba(220, 38, 38, 0.12); }
.chip-blue   { background: var(--primary-bg); border-color: var(--primary-border); }
.chip-lavender{ background: var(--gray-50);   border-color: var(--border-light); }

.chip-label { font-size: 10px; color: var(--text-tertiary); font-weight: 600; }
.chip-val {
  font-family: var(--font-ui);
  font-size: 17px; font-weight: 700; color: var(--text-primary); line-height: 1.1;
}
.chip-unit { font-size: 11px; font-weight: 500; color: var(--text-tertiary); font-family: var(--font-ui); }
.chip-sub  { font-size: 10px; color: var(--text-tertiary); }

/* Perimeter row */
.perim-row {
  display: flex; align-items: center; gap: 0;
  background: var(--gray-50);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-md);
  padding: 8px 12px;
}
.perim-item { flex: 1; display: flex; flex-direction: column; gap: 2px; }
.perim-label { font-size: 10px; color: var(--text-tertiary); }
.perim-val {
  font-family: var(--font-data);
  font-size: 12px; font-weight: 600; color: var(--text-primary);
}
.perim-divider { width: 1px; height: 28px; background: var(--border-light); margin: 0 10px; flex-shrink: 0; }

/* Chart */
.chart-wrap { height: 150px; }

.no-data {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 10px; padding: 28px;
  background: var(--gray-50);
  border: 1px dashed var(--border-light);
  border-radius: var(--radius-md);
  font-size: 12.5px; color: var(--text-disabled); text-align: center; line-height: 1.5;
}
</style>
