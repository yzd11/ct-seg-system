export const sliceUrl = (caseId, idx, center = 50, width = 400) =>
  `/api/v1/nifti/${caseId}/slice/${idx}?center=${center}&width=${width}`

export const getMetadata = (caseId) =>
  import('./client.js').then(m => m.default.get(`/nifti/${caseId}/metadata`))
