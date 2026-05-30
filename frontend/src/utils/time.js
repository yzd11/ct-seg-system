/**
 * Parse ISO 8601 string (may lack 'Z' suffix from SQLite) as UTC.
 * Returns Date object in local time zone.
 */
export function utcParse(isoStr) {
  if (!isoStr) return null
  const s = isoStr.endsWith('Z') || isoStr.includes('+') || isoStr.includes('T') === false
    ? isoStr
    : isoStr + 'Z'
  return new Date(s)
}

/**
 * Format a UTC-parsed date to relative time string (中文).
 * e.g. "刚刚", "3 分钟前", "2 小时前", "2025-05-14"
 */
export function relativeTime(date) {
  if (!date) return ''
  const d = date instanceof Date ? date : utcParse(date)
  if (!d) return ''
  const diffMs = Date.now() - d.getTime()
  const diffSec = Math.floor(diffMs / 1000)
  if (diffSec < 60) return '刚刚'
  const diffMin = Math.floor(diffSec / 60)
  if (diffMin < 60) return `${diffMin} 分钟前`
  const diffHr = Math.floor(diffMin / 60)
  if (diffHr < 24) return `${diffHr} 小时前`
  const diffDay = Math.floor(diffHr / 24)
  if (diffDay < 7) return `${diffDay} 天前`
  // fallback: YYYY-MM-DD
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const dd = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${dd}`
}
