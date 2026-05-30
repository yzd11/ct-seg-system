import { defineStore } from 'pinia'
import { ref } from 'vue'
import { listCases, getCase, deleteCase } from '../api/cases.js'

export const useCaseStore = defineStore('case', () => {
  const cases = ref([])
  const activeCase = ref(null)

  async function fetchCases() {
    try {
      const { data } = await listCases()
      cases.value = data
    } catch {
      // error handled by client.js interceptor
    }
  }

  async function setActiveCase(id) {
    try {
      const { data } = await getCase(id)
      activeCase.value = data
    } catch {
      // error handled by client.js interceptor
    }
  }

  async function removeCase(id) {
    await deleteCase(id)   // 调用方自行 catch（HomeView 已有 try/catch）
    cases.value = cases.value.filter(c => c.id !== id)
    if (activeCase.value?.id === id) activeCase.value = null
  }

  return { cases, activeCase, fetchCases, setActiveCase, removeCase }
})
