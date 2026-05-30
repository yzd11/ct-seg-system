/**
 * 模型枚举 — 单一来源。
 * 新增/删除模型只需修改这一个文件。
 */
export const MODEL_LIST = [
  { label: '① U-Net', value: 'unet' },
  { label: '② ResU-Net', value: 'resunet' },
  { label: '③ U-Net++', value: 'unet_pp' },
  { label: '④ Att. U-Net++', value: 'att_unet_pp' },
]

export const MODEL_MAP = Object.fromEntries(MODEL_LIST.map((m) => [m.value, m.label]))

export const MODEL_COLORS = {
  unet: '#3b82f6',
  resunet: '#8b5cf6',
  unet_pp: '#f59e0b',
  att_unet_pp: '#ef4444',
}
