import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/',          name: 'Home',      component: () => import('../views/HomeView.vue') },
  { path: '/upload',    name: 'Upload',    component: () => import('../views/UploadView.vue') },
  { path: '/viewer/:caseId', name: 'Viewer', component: () => import('../views/ViewerView.vue') },
  { path: '/compare',   name: 'Compare',   component: () => import('../views/CompareView.vue') },
  { path: '/history',   name: 'History',   component: () => import('../views/HistoryView.vue') },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
