<template>
  <div class="upload-view">

    <!-- Page header -->
    <PageHeader title="上传 CT 文件" desc="支持 NIfTI 格式（.nii / .nii.gz），上传后可立即进行 AI 分割推理" icon="UploadFilled" />

    <!-- Upload card -->
    <div class="upload-layout">

      <!-- Left: form -->
      <div class="form-card glass-card">
        <div class="form-card-title">
          <el-icon style="color:#2563eb"><Document /></el-icon>
          案例信息
        </div>

        <div class="form-field">
          <label class="field-label">患者 ID <span class="required">*</span></label>
          <el-input v-model="form.patientId" placeholder="例：P001" size="large" />
        </div>

        <div class="form-field">
          <label class="field-label">备注说明</label>
          <el-input v-model="form.notes" type="textarea" :rows="3" placeholder="可填写临床信息、扫描参数等..." />
        </div>

        <div class="form-field">
          <label class="field-label">NIfTI 文件 <span class="required">*</span></label>
          <el-upload
            class="ct-upload"
            drag
            :auto-upload="false"
            :on-change="onFileChange"
            accept=".nii,.nii.gz"
            :limit="1"
          >
            <div class="upload-content">
              <div class="upload-icon-wrap">
                <el-icon class="upload-main-icon"><UploadFilled /></el-icon>
                <div class="upload-ring upload-ring-1" />
                <div class="upload-ring upload-ring-2" />
              </div>
              <div class="upload-text">拖拽文件到此处</div>
              <div class="upload-hint">或 <em style="color:#2563eb;font-style:normal;font-weight:600">点击选择文件</em></div>
              <div class="upload-format">仅支持 <code>.nii</code> / <code>.nii.gz</code> 格式</div>
            </div>
          </el-upload>
        </div>

        <!-- Upload progress -->
        <div v-if="uploadPercent > 0 && uploadPercent < 100" class="upload-progress">
          <div class="progress-header">
            <span class="progress-label">上传中...</span>
            <span class="progress-pct">{{ uploadPercent }}%</span>
          </div>
          <el-progress :percentage="uploadPercent" :stroke-width="8" :show-text="false" />
        </div>

        <el-button
          type="primary"
          size="large"
          :loading="uploading"
          class="submit-btn"
          @click="submit"
        >
          <el-icon v-if="!uploading" style="margin-right:8px"><Upload /></el-icon>
          {{ uploading ? '上传中...' : '开始上传' }}
        </el-button>
      </div>

      <!-- Right: tips -->
      <div class="tips-col">
        <!-- Format info card -->
        <div class="tip-card glass-card">
          <div class="tip-card-title">
            <el-icon style="color:#0d9488"><InfoFilled /></el-icon>
            支持格式说明
          </div>
          <div class="tip-item">
            <div class="tip-dot tip-dot-blue" />
            <div>
              <div class="tip-name">.nii</div>
              <div class="tip-desc">未压缩 NIfTI 格式，体积较大但兼容性最好</div>
            </div>
          </div>
          <div class="tip-item">
            <div class="tip-dot tip-dot-teal" />
            <div>
              <div class="tip-name">.nii.gz</div>
              <div class="tip-desc">GZIP 压缩 NIfTI，推荐使用，节省存储空间</div>
            </div>
          </div>
        </div>

        <!-- Pipeline card -->
        <div class="tip-card glass-card">
          <div class="tip-card-title">
            <el-icon style="color:#6366f1"><Operation /></el-icon>
            分析流程
          </div>
          <div class="pipeline">
            <div v-for="(step, i) in pipeline" :key="i" class="pipeline-step">
              <div class="pipeline-num" :style="{ background: step.color }">{{ i + 1 }}</div>
              <div>
                <div class="pipeline-name">{{ step.name }}</div>
                <div class="pipeline-desc">{{ step.desc }}</div>
              </div>
              <div v-if="i < pipeline.length - 1" class="pipeline-line" />
            </div>
          </div>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { uploadCase } from '../api/cases.js'
import { useCaseStore } from '../stores/caseStore.js'
import PageHeader from '../components/common/PageHeader.vue'
import { ElMessage } from 'element-plus'

const router = useRouter()
const caseStore = useCaseStore()

const form = ref({ patientId: '', notes: '' })
const file = ref(null)
const uploading = ref(false)
const uploadPercent = ref(0)

const pipeline = [
  { name: '上传 CT 文件', desc: '将 NIfTI 文件传输到服务器', color: 'linear-gradient(135deg,#2563eb,#60a5fa)' },
  { name: '预处理切片',   desc: '解析体素数据并切分切片', color: 'linear-gradient(135deg,#0d9488,#5eead4)' },
  { name: 'AI 分割推理', desc: '深度学习模型逐切片推理', color: 'linear-gradient(135deg,#6366f1,#a5b4fc)' },
  { name: '生成可视化',  desc: '渲染分割遮罩与体积统计', color: 'linear-gradient(135deg,#0891b2,#38bdf8)' },
]

function onFileChange(f) {
  file.value = f.raw
  uploadPercent.value = 0
}

async function submit() {
  if (!form.value.patientId) { ElMessage.warning('请填写患者 ID'); return }
  if (!file.value) { ElMessage.warning('请选择文件'); return }
  uploading.value = true
  uploadPercent.value = 0
  try {
    const { data } = await uploadCase(
      file.value,
      form.value.patientId,
      form.value.notes,
      pct => { uploadPercent.value = pct },
    )
    uploadPercent.value = 100
    await caseStore.fetchCases()
    ElMessage.success('上传成功，正在跳转到查看器...')
    router.push(`/viewer/${data.id}`)
  } catch {
    // error handled by client.js interceptor
    uploadPercent.value = 0
  } finally {
    uploading.value = false
  }
}
</script>

<style scoped>
.upload-view { animation: page-fade-enter-active 0.3s ease both; }

/* Layout */
.upload-layout {
  display: grid;
  grid-template-columns: 1fr 340px;
  gap: 20px;
  align-items: start;
}

/* Form card */
.form-card {
  padding: 28px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.form-card-title {
  display: flex; align-items: center; gap: 8px;
  font-family: var(--font-ui);
  font-size: 16px; font-weight: 700;
  color: var(--text-primary);
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border-light);
}
.form-field { display: flex; flex-direction: column; gap: 8px; }
.field-label { font-size: 13px; font-weight: 600; color: var(--text-secondary); }
.required { color: var(--danger); margin-left: 2px; }

/* Upload zone */
.ct-upload { width: 100%; }
.upload-content {
  display: flex; flex-direction: column; align-items: center;
  padding: 32px 20px;
  gap: 8px;
}
.upload-icon-wrap {
  width: 64px; height: 64px;
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 8px;
}
.upload-main-icon {
  font-size: 36px;
  color: var(--primary);
}
.upload-text   { font-size: 15px; font-weight: 600; color: var(--text-primary); }
.upload-hint   { font-size: 13px; color: var(--text-tertiary); }
.upload-format { font-size: 12px; color: var(--text-disabled); }
.upload-format code {
  background: var(--gray-100);
  padding: 1px 6px;
  border-radius: 4px;
  color: var(--primary);
  font-family: var(--font-data);
  font-size: 11px;
}

/* Progress */
.upload-progress { display: flex; flex-direction: column; gap: 8px; }
.progress-header { display: flex; justify-content: space-between; }
.progress-label { font-size: 13px; color: var(--text-secondary); font-weight: 500; }
.progress-pct   { font-size: 13px; font-weight: 700; color: var(--primary); font-family: var(--font-data); }

/* Submit button */
.submit-btn { width: 100%; height: 48px; font-size: 15px; font-weight: 600; }

/* ── Right tips column ──────────────────────────────────── */
.tips-col { display: flex; flex-direction: column; gap: 16px; }

.tip-card { padding: 20px 22px; display: flex; flex-direction: column; gap: 14px; }
.tip-card-title {
  display: flex; align-items: center; gap: 8px;
  font-family: var(--font-ui);
  font-size: 15px; font-weight: 700; color: var(--text-primary);
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border-light);
}

.tip-item { display: flex; align-items: flex-start; gap: 12px; }
.tip-dot {
  width: 8px; height: 8px;
  border-radius: 50%; flex-shrink: 0;
  margin-top: 5px;
}
.tip-dot-blue { background: var(--primary); }
.tip-dot-teal { background: var(--success); }
.tip-name { font-size: 13px; font-weight: 600; color: var(--text-primary); font-family: var(--font-data); }
.tip-desc { font-size: 12px; color: var(--text-tertiary); margin-top: 2px; line-height: 1.5; }

/* Pipeline */
.pipeline { display: flex; flex-direction: column; gap: 0; }
.pipeline-step {
  display: flex; align-items: flex-start; gap: 12px;
  position: relative; padding-bottom: 16px;
}
.pipeline-step:last-child { padding-bottom: 0; }
.pipeline-num {
  width: 26px; height: 26px;
  border-radius: 50%;
  color: #fff;
  font-size: 12px;
  font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.pipeline-name { font-size: 13px; font-weight: 600; color: var(--text-primary); }
.pipeline-desc { font-size: 11.5px; color: var(--text-tertiary); margin-top: 2px; }
.pipeline-line {
  position: absolute;
  left: 12px;
  top: 28px; bottom: 0;
  width: 2px;
  background: var(--border-light);
}
</style>
