<template>
  <div class="home">

    <PageHeader title="案例列表" desc="管理所有已上传的 CT 影像案例，点击卡片进入查看器" icon="Folder">
      <el-button type="primary" size="large" @click="$router.push('/upload')">
        <el-icon style="margin-right:6px"><Upload /></el-icon>
        上传新案例
      </el-button>
    </PageHeader>

    <!-- ── Stat cards ───────────────────────────────────────── -->
    <div class="stat-row">
      <div class="stat-card strip-blue">
        <div class="stat-icon-wrap" style="background:linear-gradient(135deg,#eff6ff,#dbeafe)">
          <el-icon style="color:#2563eb;font-size:22px"><Folder /></el-icon>
        </div>
        <div class="stat-body">
          <div class="stat-num" style="color:#1d4ed8">{{ animTotal }}</div>
          <div class="stat-label">总案例数</div>
          <div class="stat-sub">All Cases</div>
        </div>
        <div class="stat-bg-icon" style="color:rgba(37,99,235,0.06)">
          <el-icon style="font-size:80px"><Folder /></el-icon>
        </div>
      </div>

      <div class="stat-card strip-teal">
        <div class="stat-icon-wrap" style="background:linear-gradient(135deg,#f0fdfa,#ccfbf1)">
          <el-icon style="color:#0d9488;font-size:22px"><UploadFilled /></el-icon>
        </div>
        <div class="stat-body">
          <div class="stat-num" style="color:#0d9488">{{ animToday }}</div>
          <div class="stat-label">今日上传</div>
          <div class="stat-sub">Today</div>
        </div>
        <div class="stat-bg-icon" style="color:rgba(13,148,136,0.06)">
          <el-icon style="font-size:80px"><UploadFilled /></el-icon>
        </div>
      </div>

      <div class="stat-card strip-lavender">
        <div class="stat-icon-wrap" style="background:linear-gradient(135deg,#eef2ff,#e0e7ff)">
          <el-icon style="color:#6366f1;font-size:22px"><Clock /></el-icon>
        </div>
        <div class="stat-body">
          <div class="stat-num" style="color:#4f46e5">{{ animRecent }}</div>
          <div class="stat-label">近 7 天新增</div>
          <div class="stat-sub">Last 7 Days</div>
        </div>
        <div class="stat-bg-icon" style="color:rgba(99,102,241,0.06)">
          <el-icon style="font-size:80px"><Clock /></el-icon>
        </div>
      </div>
    </div>

    <!-- ── Empty state ──────────────────────────────────────── -->
    <div v-if="!cases.length" class="empty-wrap">
      <div class="empty-inner">
        <div class="empty-icon">
          <el-icon style="font-size:52px;color:#d1d5db"><UploadFilled /></el-icon>
        </div>
        <div class="empty-title">暂无案例</div>
        <div class="empty-desc">点击右上角「上传新案例」按钮，上传 NIfTI 格式的 CT 影像开始分析</div>
        <el-button type="primary" @click="$router.push('/upload')" style="margin-top:20px">
          <el-icon style="margin-right:6px"><Upload /></el-icon>立即上传
        </el-button>
      </div>
    </div>

    <!-- ── Case grid ────────────────────────────────────────── -->
    <div v-else>
      <!-- Section header -->
      <div class="list-header">
        <div class="list-header-left">
          <span class="section-title">影像案例</span>
          <span class="case-count">共 {{ cases.length }} 条</span>
        </div>
      </div>

      <div class="case-grid">
        <div
          v-for="c in pagedCases"
          :key="c.id"
          class="case-card"
          @click="openViewer(c.id)"
        >
          <!-- Color top strip -->
          <div class="card-strip" :style="{ background: avatarGradient(c.patient_id) }" />

          <!-- Card body -->
          <div class="card-body">
            <div class="card-header-row">
              <div class="patient-avatar" :style="{ background: avatarGradient(c.patient_id) }">
                {{ c.patient_id?.charAt(0)?.toUpperCase() || 'P' }}
              </div>
              <div class="patient-meta">
                <div class="patient-id">{{ c.patient_id }}</div>
                <div class="patient-file" :title="c.filename">{{ c.filename }}</div>
              </div>
              <div class="card-arrow">
                <el-icon><ArrowRight /></el-icon>
              </div>
            </div>

            <div class="card-tags">
              <span class="tag tag-blue">
                <el-icon style="margin-right:3px;font-size:11px"><Grid /></el-icon>
                {{ c.slice_count }} 切片
              </span>
              <span class="tag tag-teal">CT 影像</span>
            </div>

            <div class="card-time">
              <el-icon style="margin-right:4px;font-size:12px;color:#94afc8"><Clock /></el-icon>
              {{ formatTime(c.upload_time) }}
            </div>
          </div>

          <!-- Card footer -->
          <div class="card-footer">
            <el-button size="small" type="primary" @click.stop="openViewer(c.id)">
              <el-icon style="margin-right:4px"><View /></el-icon>查看
            </el-button>
            <el-button size="small" type="danger" @click.stop="remove(c.id)">
              <el-icon style="margin-right:4px"><Delete /></el-icon>删除
            </el-button>
          </div>
        </div>
      </div>

      <!-- Pagination -->
      <div v-if="cases.length > PAGE_SIZE" class="pagination-wrap">
        <el-pagination
          v-model:current-page="currentPage"
          :page-size="PAGE_SIZE"
          :total="cases.length"
          layout="total, prev, pager, next"
          background
        />
      </div>
    </div>

  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { storeToRefs } from 'pinia'
import { useCaseStore } from '../stores/caseStore.js'
import { useViewerStore } from '../stores/viewerStore.js'
import PageHeader from '../components/common/PageHeader.vue'
import { ElMessageBox, ElMessage } from 'element-plus'

const caseStore = useCaseStore()
const viewer = useViewerStore()
const router = useRouter()
const { cases } = storeToRefs(caseStore)

const currentPage = ref(1)
const PAGE_SIZE = 15

const pagedCases = computed(() => {
  const start = (currentPage.value - 1) * PAGE_SIZE
  return cases.value.slice(start, start + PAGE_SIZE)
})

// Reset to page 1 when cases change (e.g. after delete or reload)
watch(() => cases.value.length, () => { currentPage.value = 1 })


onMounted(() => caseStore.fetchCases())

// ── Computed counts ───────────────────────────────────────
const todayCount = computed(() => {
  const today = new Date().toDateString()
  return cases.value.filter(c => c.upload_time && new Date(c.upload_time).toDateString() === today).length
})
const recentCount = computed(() => {
  const cutoff = Date.now() - 7 * 24 * 60 * 60 * 1000
  return cases.value.filter(c => c.upload_time && new Date(c.upload_time).getTime() > cutoff).length
})

// ── Animated counters ─────────────────────────────────────
const animTotal  = ref(0)
const animToday  = ref(0)
const animRecent = ref(0)

function animateTo(refVal, target, duration = 800) {
  const start = performance.now()
  const from = refVal.value
  function step(now) {
    const p = Math.min((now - start) / duration, 1)
    const eased = 1 - Math.pow(1 - p, 3)
    refVal.value = Math.round(from + (target - from) * eased)
    if (p < 1) requestAnimationFrame(step)
  }
  requestAnimationFrame(step)
}

watch(() => cases.value.length,  v => animateTo(animTotal,  v), { immediate: true })
watch(todayCount,  v => animateTo(animToday,  v), { immediate: true })
watch(recentCount, v => animateTo(animRecent, v), { immediate: true })

// ── Gradients ─────────────────────────────────────────────
const gradients = [
  'linear-gradient(135deg, #2563eb, #60a5fa)',
  'linear-gradient(135deg, #0d9488, #5eead4)',
  'linear-gradient(135deg, #6366f1, #a5b4fc)',
  'linear-gradient(135deg, #0891b2, #38bdf8)',
  'linear-gradient(135deg, #059669, #6ee7b7)',
  'linear-gradient(135deg, #7c3aed, #c4b5fd)',
]
function avatarGradient(pid) {
  return gradients[(pid?.charCodeAt(0) ?? 0) % gradients.length]
}

function openViewer(id) {
  viewer.activeJobId = null
  router.push(`/viewer/${id}`)
}

async function remove(id) {
  try {
    await ElMessageBox.confirm('确认删除该案例及所有推理结果？', '删除确认', { type: 'warning' })
  } catch { return }
  try {
    await caseStore.removeCase(id)
    ElMessage.success('已删除')
  } catch {
    // error handled by client.js interceptor
  }
}

function formatTime(t) {
  return t ? new Date(t).toLocaleString('zh-CN') : ''
}
</script>

<style scoped>
.home { animation: page-fade-enter-active 0.3s ease both; }

/* ── Stat cards ───────────────────────────────────────────── */
.stat-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-bottom: 28px;
}
.stat-card {
  position: relative;
  background: var(--bg-card);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-lg);
  padding: 22px 20px;
  display: flex;
  align-items: center;
  gap: 18px;
  box-shadow: var(--shadow-sm);
  overflow: hidden;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  cursor: default;
}
.stat-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.stat-icon-wrap {
  width: 52px; height: 52px;
  border-radius: var(--radius-lg);
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.stat-body { flex: 1; }
.stat-num {
  font-family: var(--font-ui);
  font-size: 34px;
  font-weight: 700;
  line-height: 1;
  color: var(--text-primary);
}
.stat-label { font-size: 13px; color: var(--text-secondary); font-weight: 500; margin-top: 4px; }
.stat-sub   { font-size: 10px; color: var(--text-tertiary); font-family: var(--font-data); margin-top: 2px; }

.stat-bg-icon {
  position: absolute;
  right: -10px; bottom: -10px;
  pointer-events: none;
  line-height: 1;
  opacity: 0.04;
}

/* ── Empty state ──────────────────────────────────────────── */
.empty-wrap {
  display: flex; justify-content: center; padding: 60px 0;
}
.empty-inner {
  text-align: center;
  background: var(--bg-card);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-xl);
  padding: 48px 56px;
  box-shadow: var(--shadow-sm);
}
.empty-icon { margin-bottom: 16px; }
.empty-title { font-family: var(--font-ui); font-size: 20px; font-weight: 700; color: var(--text-primary); margin-bottom: 8px; }
.empty-desc  { font-size: 13px; color: var(--text-tertiary); max-width: 320px; line-height: 1.6; }

/* ── List header ──────────────────────────────────────────── */
.list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
}
.list-header-left { display: flex; align-items: center; gap: 10px; }
.case-count {
  background: var(--gray-100);
  color: var(--text-secondary);
  border: 1px solid var(--border-light);
  border-radius: 99px;
  padding: 2px 10px;
  font-size: 12px;
  font-weight: 500;
}

/* ── Case grid ────────────────────────────────────────────── */
.case-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(270px, 1fr));
  gap: 16px;
}

/* ── Case card ────────────────────────────────────────────── */
.case-card {
  background: var(--bg-card);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-lg);
  overflow: hidden;
  cursor: pointer;
  display: flex; flex-direction: column;
  box-shadow: var(--shadow-xs);
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.case-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
  border-color: var(--gray-300);
}

.card-strip { height: 4px; flex-shrink: 0; }

.card-body { padding: 16px 16px 12px; flex: 1; }

.card-header-row {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 12px;
}
.patient-avatar {
  width: 42px; height: 42px;
  border-radius: var(--radius-lg);
  color: #fff;
  font-size: 17px;
  font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.patient-meta { flex: 1; min-width: 0; }
.patient-id {
  font-size: 15px;
  font-weight: 700;
  color: var(--text-primary);
  line-height: 1.3;
}
.patient-file {
  font-size: 11.5px;
  color: var(--text-tertiary);
  margin-top: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-family: var(--font-data);
}
.card-arrow {
  color: var(--gray-300);
  font-size: 16px;
  transition: all var(--ease);
  margin-top: 4px;
}
.case-card:hover .card-arrow { color: var(--primary); transform: translateX(3px); }

.card-tags { display: flex; gap: 6px; margin-bottom: 10px; flex-wrap: wrap; }
.tag {
  display: inline-flex; align-items: center;
  padding: 3px 9px;
  border-radius: 99px;
  font-size: 11.5px;
  font-weight: 600;
  border: 1px solid transparent;
}
.tag-blue { background: var(--primary-bg); color: var(--primary); border-color: var(--primary-border); }
.tag-teal { background: var(--success-bg); color: var(--success); border-color: rgba(5, 150, 105, 0.18); }

.card-time {
  display: flex; align-items: center;
  font-size: 11.5px;
  color: var(--text-tertiary);
  font-family: var(--font-data);
}

.card-footer {
  padding: 10px 14px 14px;
  display: flex; gap: 8px;
  border-top: 1px solid var(--gray-100);
  background: var(--gray-50);
}

.pagination-wrap {
  display: flex;
  justify-content: center;
  padding: 24px 0 8px;
}
</style>
