<template>
  <div class="viewer-layout" v-if="caseInfo">

    <!-- Left: Canvas + controls -->
    <div class="viewer-left">
      <div class="canvas-wrap glass-card">
        <div class="canvas-header">
          <div class="canvas-title">
            <el-icon style="color:#2563eb"><Monitor /></el-icon>
            CT 影像查看器
          </div>
          <div class="canvas-info">
            <span class="canvas-badge">切片 {{ viewer.currentSlice + 1 }} / {{ viewer.totalSlices }}</span>
            <span v-if="activeJob?.status === 'done'" class="canvas-badge badge-green">分割完成</span>
            <span v-else-if="activeJob?.status === 'running'" class="canvas-badge badge-amber">推理中...</span>
          </div>
        </div>
        <div class="canvas-area">
          <SliceCanvas
            :case-id="caseId"
            :slice-index="viewer.currentSlice"
            :job-id="viewer.activeJobId"
          />
        </div>
      </div>

      <div class="controls-card glass-card">
        <SliceSlider />
        <div class="controls-divider" />
        <div class="controls-row">
          <OverlayControls />
          <WindowingControls />
        </div>
      </div>
    </div>

    <!-- Right: panels + sticky export -->
    <div class="viewer-right">

      <!-- Scrollable panels area -->
      <div class="panels-scroll">

        <!-- Inference panel -->
        <div class="panel glass-card panel-blue">
          <div class="panel-header">
            <div class="panel-header-left">
              <div class="panel-icon icon-blue-sm"><el-icon><Cpu /></el-icon></div>
              <span class="panel-title">推理控制</span>
            </div>
            <span class="panel-tag">AI Engine</span>
          </div>
          <div class="panel-body">
            <ModelSelector :case-id="caseId" />
            <div v-if="activeJob" class="mt-12">
              <ProgressBar :job="activeJob" />
              <el-button
                v-if="activeJob.status === 'running'"
                size="small" type="danger" style="margin-top:10px;width:100%"
                @click="jobStore.cancel(activeJob.id)"
              >
                <el-icon style="margin-right:4px"><VideoPause /></el-icon>取消推理
              </el-button>
            </div>
          </div>
        </div>

        <!-- Metrics panel -->
        <div class="panel glass-card panel-teal">
          <div class="panel-header">
            <div class="panel-header-left">
              <div class="panel-icon icon-teal-sm"><el-icon><DataAnalysis /></el-icon></div>
              <span class="panel-title">分割指标</span>
            </div>
            <span class="panel-tag tag-teal">Metrics</span>
          </div>
          <div class="panel-body">
            <!-- Volume cards -->
            <div v-if="activeJob?.status === 'done'" class="volume-row">
              <div class="volume-card vc-liver">
                <div class="vc-emoji">🫀</div>
                <div class="vc-val">{{ activeJob.liver_volume_ml?.toFixed(1) ?? '—' }}</div>
                <div class="vc-unit">mL</div>
                <div class="vc-label">肝脏体积</div>
              </div>
              <div class="volume-card vc-tumor">
                <div class="vc-emoji">🔴</div>
                <div class="vc-val">{{ activeJob.tumor_volume_ml?.toFixed(1) ?? '—' }}</div>
                <div class="vc-unit">mL</div>
                <div class="vc-label">肿瘤体积</div>
              </div>
            </div>
            <MetricsPanel :job="activeJob" />
          </div>
        </div>

        <!-- Case info panel -->
        <div class="panel glass-card panel-lavender">
          <div class="panel-header">
            <div class="panel-header-left">
              <div class="panel-icon icon-lavender-sm"><el-icon><InfoFilled /></el-icon></div>
              <span class="panel-title">案例信息</span>
            </div>
            <span class="panel-tag tag-lavender">Info</span>
          </div>
          <div class="panel-body">
            <div class="info-table">
              <div class="info-row">
                <span class="info-key">患者 ID</span>
                <span class="info-val">{{ caseInfo.patient_id }}</span>
              </div>
              <div class="info-row">
                <span class="info-key">文件名</span>
                <span class="info-val info-file" :title="caseInfo.filename">{{ caseInfo.filename }}</span>
              </div>
              <div class="info-row">
                <span class="info-key">切片数</span>
                <span class="info-val">
                  <span class="tag tag-blue-sm">{{ caseInfo.slice_count }} 张</span>
                </span>
              </div>
              <div class="info-row">
                <span class="info-key">体素间距</span>
                <span class="info-val info-mono">{{ spacingStr }} mm</span>
              </div>
            </div>
          </div>
        </div>

      </div><!-- /panels-scroll -->

      <!-- Export button always pinned at bottom -->
      <div class="export-dock">
        <ExportButton
          :job-id="viewer.activeJobId"
          :is-done="activeJob?.status === 'done'"
        />
      </div>

    </div>
  </div>

  <!-- Loading -->
  <div v-else class="viewer-loading glass-card">
    <el-skeleton :rows="6" animated />
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { getCase } from '../api/cases.js'
import { getJob } from '../api/inference.js'
import { useViewerStore } from '../stores/viewerStore.js'
import { useJobStore } from '../stores/jobStore.js'

import SliceCanvas from '../components/viewer/SliceCanvas.vue'
import SliceSlider from '../components/viewer/SliceSlider.vue'
import OverlayControls from '../components/viewer/OverlayControls.vue'
import WindowingControls from '../components/viewer/WindowingControls.vue'
import ModelSelector from '../components/inference/ModelSelector.vue'
import ProgressBar from '../components/inference/ProgressBar.vue'
import MetricsPanel from '../components/inference/MetricsPanel.vue'
import ExportButton from '../components/export/ExportButton.vue'

const route = useRoute()
const caseId = route.params.caseId
const viewer = useViewerStore()
const jobStore = useJobStore()
const caseInfo = ref(null)

onMounted(async () => {
  const savedJobId = viewer.activeJobId  // may be set by HistoryView before navigation
  viewer.resetState()
  viewer.activeJobId = savedJobId

  const { data } = await getCase(caseId)
  caseInfo.value = data
  viewer.totalSlices = data.slice_count || 0
  viewer.currentSlice = 0
  if (viewer.activeJobId) {
    try {
      const { data: job } = await getJob(viewer.activeJobId)
      jobStore.jobs[job.id] = job
    } catch {
      viewer.activeJobId = null   // 历史 job 不存在则清除，停止无效引用
    }
  }
})

const activeJob = computed(() =>
  viewer.activeJobId ? jobStore.jobs[viewer.activeJobId] : null
)
const spacingStr = computed(() => {
  if (!caseInfo.value?.voxel_spacing) return '—'
  try {
    const s = JSON.parse(caseInfo.value.voxel_spacing)
    return s.map(v => v.toFixed(2)).join(' × ')
  } catch {
    return '—'
  }
})
</script>

<style scoped>
.viewer-layout {
  display: flex;
  gap: 16px;
  height: calc(100vh - 108px);
  animation: page-fade-enter-active 0.3s ease both;
}

/* ── Left column ──────────────────────────────────────────── */
.viewer-left {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.canvas-wrap {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding: 0;
}
.canvas-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 18px;
  border-bottom: 1px solid var(--border-light);
  flex-shrink: 0;
}
.canvas-title {
  display: flex; align-items: center; gap: 8px;
  font-family: var(--font-ui);
  font-size: 14px; font-weight: 700; color: var(--text-primary);
}
.canvas-info { display: flex; gap: 8px; }
.canvas-badge {
  padding: 3px 10px;
  border-radius: 99px;
  font-size: 11.5px;
  font-weight: 600;
  background: var(--gray-100);
  color: var(--text-secondary);
  border: 1px solid var(--border-light);
  font-family: var(--font-data);
}
.badge-green  { background: var(--success-bg); color: var(--success); border-color: rgba(5, 150, 105, 0.2); }
.badge-amber  { background: var(--warning-bg); color: var(--warning); border-color: rgba(217, 119, 6, 0.2); }

.canvas-area { flex: 1; min-height: 0; }

.controls-card {
  flex-shrink: 0;
  padding: 14px 18px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.controls-divider { height: 1px; background: var(--border-light); }
.controls-row { display: flex; gap: 16px; flex-wrap: wrap; }

/* ── Right panels ─────────────────────────────────────────── */
.viewer-right {
  width: 300px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.panels-scroll {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding-right: 2px;
}

.export-dock {
  flex-shrink: 0;
  padding: 10px 0 0;
}

.panel { padding: 0; overflow: hidden; flex-shrink: 0; }

/* Border-left color accent — unified to primary */
.panel-blue    { border-left: 3px solid var(--primary) !important; }
.panel-teal    { border-left: 3px solid var(--primary) !important; }
.panel-lavender{ border-left: 3px solid var(--primary) !important; }

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-light);
}
.panel-header-left { display: flex; align-items: center; gap: 8px; }

.panel-icon {
  width: 28px; height: 28px;
  border-radius: var(--radius-sm);
  display: flex; align-items: center; justify-content: center;
  font-size: 14px;
  background: var(--primary-bg); color: var(--primary);
}

.panel-title {
  font-family: var(--font-ui);
  font-size: 14px; font-weight: 700; color: var(--text-primary);
}

.panel-tag {
  padding: 2px 8px;
  border-radius: 99px;
  font-size: 10px; font-weight: 600;
  font-family: var(--font-data);
  background: var(--gray-100); color: var(--text-tertiary);
  border: 1px solid var(--border-light);
}
.tag-teal    { background: var(--success-bg); color: var(--success); border-color: rgba(5, 150, 105, 0.2); }
.tag-lavender{ background: var(--primary-bg); color: var(--primary); border-color: var(--primary-border); }

.panel-body { padding: 14px 16px; }
.mt-12 { margin-top: 12px; }

/* Volume cards */
.volume-row { display: flex; gap: 10px; margin-bottom: 14px; }
.volume-card {
  flex: 1; border-radius: var(--radius-md); padding: 12px 10px;
  text-align: center; border: 1px solid transparent;
}
.vc-liver { background: var(--success-bg); border-color: rgba(5, 150, 105, 0.15); }
.vc-tumor { background: var(--danger-bg); border-color: rgba(220, 38, 38, 0.15); }
.vc-emoji { font-size: 18px; margin-bottom: 4px; }
.vc-val   { font-family: var(--font-ui); font-size: 22px; font-weight: 700; color: var(--text-primary); line-height: 1; }
.vc-unit  { font-size: 11px; color: var(--text-tertiary); margin-top: 1px; }
.vc-label { font-size: 12px; color: var(--text-secondary); margin-top: 4px; }

/* Info table */
.info-table { display: flex; flex-direction: column; gap: 0; }
.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid var(--gray-100);
  font-size: 13px;
}
.info-row:last-child { border-bottom: none; }
.info-key  { color: var(--text-tertiary); font-weight: 500; }
.info-val  { color: var(--text-primary); font-weight: 600; text-align: right; }
.info-file { font-size: 11.5px; max-width: 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: var(--font-data); color: var(--text-secondary); }
.info-mono { font-family: var(--font-data); font-size: 12px; }

.tag-blue-sm {
  background: var(--primary-bg); color: var(--primary);
  border: 1px solid var(--primary-border);
  border-radius: var(--radius-sm); padding: 2px 8px;
  font-size: 12px; font-weight: 600;
}

.viewer-loading { padding: 40px; }
</style>
