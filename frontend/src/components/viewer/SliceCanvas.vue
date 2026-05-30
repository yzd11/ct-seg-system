<template>
  <div class="canvas-wrapper" ref="wrapperRef" @wheel.prevent="onWheel">
    <canvas ref="canvasRef" class="ct-canvas" />
    <div v-if="loading" class="canvas-loading">
      <el-icon class="is-loading"><Loading /></el-icon>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted, onBeforeUnmount } from 'vue'
import { ElMessage } from 'element-plus'
import { useViewerStore } from '../../stores/viewerStore.js'
import { sliceUrl } from '../../api/nifti.js'
import { maskUrl } from '../../api/inference.js'

const props = defineProps({
  caseId: { type: String, required: true },
  sliceIndex: { type: Number, required: true },
  jobId: { type: String, default: null },
})

const viewer = useViewerStore()
const canvasRef = ref(null)
const wrapperRef = ref(null)
const loading = ref(false)

let ctImg = null
let overlayImg = null
let _loadSeq = 0   // monotonically increasing, prevents stale responses from overwriting

// ── LRU slice cache ────────────────────────────────────────
const CACHE_LIMIT = 50
const _ctCache = new Map()       // key → HTMLImageElement
const _maskCache = new Map()

function _cacheKey(idx) {
  return `${props.caseId}|${props.jobId || 'nomask'}|${idx}`
}

function _evictLru(map) {
  if (map.size >= CACHE_LIMIT) {
    const first = map.keys().next().value
    map.delete(first)
  }
}

function invalidateCache(newCaseId, newJobId) {
  // Clear ct cache if case changed (window params may differ per case)
  if (newCaseId && newCaseId !== _lastCaseId) {
    _ctCache.clear()
    _lastCaseId = newCaseId
  }
  // Mask cache cleared if jobId changed
  if (newJobId !== undefined && newJobId !== _lastJobId) {
    _maskCache.clear()
    _lastJobId = newJobId
  }
}
let _lastCaseId = null
let _lastJobId = null

// Cached separated layer canvases (rebuilt on mask load or canvas resize)
let _liverCanvas = null
let _tumorCanvas = null
let _layerW = 0
let _layerH = 0

/**
 * Split a combined RGBA mask image into two separate offscreen canvases:
 * one for liver (green-dominant pixels) and one for tumor (red-dominant).
 */
function _buildLayers(img, w, h) {
  const tmp = document.createElement('canvas')
  tmp.width = w; tmp.height = h
  tmp.getContext('2d').drawImage(img, 0, 0, w, h)
  const raw = tmp.getContext('2d').getImageData(0, 0, w, h).data

  const liverData = new ImageData(w, h)
  const tumorData = new ImageData(w, h)
  const ld = liverData.data
  const td = tumorData.data

  for (let i = 0; i < raw.length; i += 4) {
    const r = raw[i], g = raw[i + 1], b = raw[i + 2], a = raw[i + 3]
    if (a === 0) continue
    if (g > r + 40 && g > b + 40) {
      // Liver: green-dominant (stored as (0,200,0,160))
      ld[i] = r; ld[i + 1] = g; ld[i + 2] = b; ld[i + 3] = a
    } else if (r > g + 40 && r > b + 40) {
      // Tumor: red-dominant (stored as (220,30,30,200))
      td[i] = r; td[i + 1] = g; td[i + 2] = b; td[i + 3] = a
    }
  }

  _liverCanvas = document.createElement('canvas')
  _liverCanvas.width = w; _liverCanvas.height = h
  _liverCanvas.getContext('2d').putImageData(liverData, 0, 0)

  _tumorCanvas = document.createElement('canvas')
  _tumorCanvas.width = w; _tumorCanvas.height = h
  _tumorCanvas.getContext('2d').putImageData(tumorData, 0, 0)

  _layerW = w; _layerH = h
}

function drawCanvas() {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  const { width, height } = canvas

  ctx.clearRect(0, 0, width, height)

  if (ctImg) {
    ctx.globalAlpha = 1
    ctx.drawImage(ctImg, 0, 0, width, height)
  }

  if (overlayImg) {
    // Rebuild layer caches if mask changed or canvas resized
    if (!_liverCanvas || _layerW !== width || _layerH !== height) {
      _buildLayers(overlayImg, width, height)
    }
    ctx.globalAlpha = viewer.overlayOpacity
    if (viewer.showLiver && _liverCanvas) ctx.drawImage(_liverCanvas, 0, 0)
    if (viewer.showTumor && _tumorCanvas) ctx.drawImage(_tumorCanvas, 0, 0)
    ctx.globalAlpha = 1
  }
}

function _ctUrl(idx) {
  return sliceUrl(props.caseId, idx, viewer.windowCenter, viewer.windowWidth)
}
function _maskUrl(idx) {
  return props.jobId ? maskUrl(props.jobId, idx) : null
}

async function loadImages() {
  const seq = ++_loadSeq
  loading.value = true
  const idx = props.sliceIndex

  // Try cache first
  const ck = _cacheKey(idx)
  const cachedCt = _ctCache.get(ck)
  const cachedMask = _maskCache.get(ck)

  if (cachedCt) {
    ctImg = cachedCt
    overlayImg = cachedMask || null
    if (overlayImg) { _liverCanvas = null; _tumorCanvas = null }
    loading.value = false
    drawCanvas()

    // Background prefetch neighbours
    _prefetch(idx - 2); _prefetch(idx - 1)
    _prefetch(idx + 1); _prefetch(idx + 2)
    return
  }

  // Fallback: network load
  const ctUrl = _ctUrl(idx)
  const ovUrl = _maskUrl(idx)

  const [newCt, newOv] = await Promise.all([
    loadImg(ctUrl, 'CT'),
    ovUrl ? loadImg(ovUrl, 'mask') : Promise.resolve(null),
  ])

  if (seq !== _loadSeq) return

  if (newCt) {
    _evictLru(_ctCache)
    _ctCache.set(ck, newCt)
  }
  if (newOv) {
    _evictLru(_maskCache)
    _maskCache.set(ck, newOv)
  }

  ctImg = newCt || ctImg
  overlayImg = newOv
  _liverCanvas = null
  _tumorCanvas = null
  loading.value = false
  drawCanvas()

  // Background prefetch
  _prefetch(idx - 2); _prefetch(idx - 1)
  _prefetch(idx + 1); _prefetch(idx + 2)
}

async function _prefetch(idx) {
  if (idx < 0) return
  const ck = _cacheKey(idx)
  if (_ctCache.has(ck)) return
  try {
    const ctUrl = _ctUrl(idx)
    const ovUrl = _maskUrl(idx)
    const [ct, ov] = await Promise.all([
      loadImg(ctUrl, 'CT'),
      ovUrl ? loadImg(ovUrl, 'mask') : Promise.resolve(null),
    ])
    if (ct) { _evictLru(_ctCache); _ctCache.set(ck, ct) }
    if (ov) { _evictLru(_maskCache); _maskCache.set(ck, ov) }
  } catch { /* prefetch failure is silent */ }
}

function loadImg(src, label) {
  return new Promise((res) => {
    const img = new Image()
    img.onload = () => res(img)
    img.onerror = () => {
      if (label === 'CT') {
        ElMessage.warning(`切片 ${props.sliceIndex} 加载失败`)
      }
      res(null)
    }
    img.src = src
  })
}

// Resize canvas to match wrapper using ResizeObserver
let _ro = null
function resizeCanvas() {
  const canvas = canvasRef.value
  const wrapper = wrapperRef.value
  if (!canvas || !wrapper) return
  canvas.width = wrapper.clientWidth
  canvas.height = wrapper.clientHeight
  drawCanvas()
}

// Scroll wheel for slice navigation
function onWheel(e) {
  if (e.deltaY > 0) {
    viewer.setSlice(viewer.currentSlice + 1)
  } else {
    viewer.setSlice(viewer.currentSlice - 1)
  }
}

// Keyboard navigation — only when no input element is focused
function onKeydown(e) {
  const tag = document.activeElement?.tagName?.toLowerCase()
  if (tag === 'input' || tag === 'textarea' || document.activeElement?.isContentEditable) return
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
    viewer.setSlice(viewer.currentSlice + 1)
  } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
    viewer.setSlice(viewer.currentSlice - 1)
  }
}

onMounted(() => {
  resizeCanvas()
  _ro = new ResizeObserver(resizeCanvas)
  _ro.observe(wrapperRef.value)
  window.addEventListener('keydown', onKeydown)
  loadImages()
})

onBeforeUnmount(() => {
  _ro?.disconnect()
  window.removeEventListener('keydown', onKeydown)
})

// 调窗防抖：300ms 内只触发一次，且仅清 CT 缓存
let _windowDebounceTimer = null
function debouncedLoad() {
  if (_windowDebounceTimer) clearTimeout(_windowDebounceTimer)
  _windowDebounceTimer = setTimeout(() => {
    _ctCache.clear()   // window params changed → CT images stale
    _lastCaseId = null
    loadImages()
  }, 300)
}

// 切片/job 变化立即加载；调窗变化走防抖
watch(() => [props.sliceIndex, props.jobId], () => {
  invalidateCache(props.caseId, props.jobId)
  loadImages()
})
watch(() => [viewer.windowCenter, viewer.windowWidth], () => debouncedLoad())
watch(
  () => [viewer.showLiver, viewer.showTumor, viewer.overlayOpacity],
  () => drawCanvas(),
)
</script>

<style scoped>
.canvas-wrapper {
  position: relative;
  width: 100%;
  height: 100%;
  background: #000;
  border-radius: 8px;
  overflow: hidden;
  cursor: crosshair;
}
.ct-canvas {
  display: block;
  width: 100%;
  height: 100%;
}
.canvas-loading {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0,0,0,0.5);
  color: #fff;
  font-size: 32px;
}
</style>
