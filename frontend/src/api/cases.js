import client from './client.js'

export const listCases = () => client.get('/cases/')
export const getCase = (id) => client.get(`/cases/${id}`)
export const deleteCase = (id) => client.delete(`/cases/${id}`)

export const uploadCase = (file, patientId, notes = '', onProgress) => {
  const form = new FormData()
  form.append('file', file)
  form.append('patient_id', patientId)
  form.append('notes', notes)
  return client.post('/cases/', form, {
    onUploadProgress: onProgress
      ? e => onProgress(e.total ? Math.round((e.loaded / e.total) * 100) : 0)
      : undefined,
  })
}
