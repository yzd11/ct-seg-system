import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useViewerStore = defineStore('viewer', () => {
  const currentSlice = ref(0)
  const totalSlices = ref(0)
  const showLiver = ref(true)
  const showTumor = ref(true)
  const overlayOpacity = ref(0.6)
  const windowCenter = ref(50)
  const windowWidth = ref(400)
  const activeJobId = ref(null)

  function setSlice(idx) {
    currentSlice.value = Math.max(0, Math.min(idx, totalSlices.value - 1))
  }

  function resetState() {
    currentSlice.value = 0
    totalSlices.value = 0
    activeJobId.value = null
  }

  return {
    currentSlice, totalSlices, showLiver, showTumor,
    overlayOpacity, windowCenter, windowWidth, activeJobId,
    setSlice, resetState,
  }
})
