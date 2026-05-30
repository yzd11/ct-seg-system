import client from './client.js'

export const submitJob = (caseId, modelName) =>
  client.post('/inference/jobs', { case_id: caseId, model_name: modelName })

export const getJob = (jobId) => client.get(`/inference/jobs/${jobId}`)

export const getJobResults = (jobId) => client.get(`/inference/jobs/${jobId}/results`)

export const cancelJob = (jobId) => client.delete(`/inference/jobs/${jobId}`)

export const getCaseJobs = (caseId) => client.get(`/inference/cases/${caseId}/jobs`)

export const maskUrl = (jobId, idx) => `/api/v1/inference/jobs/${jobId}/mask/${idx}`
