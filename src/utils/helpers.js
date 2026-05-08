export function formatDate(dateStr) {
  const d = new Date(dateStr + 'T00:00:00');
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

export function formatDateShort(dateStr) {
  const d = new Date(dateStr + 'T00:00:00');
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

export function daysUntil(dateStr) {
  const target = new Date(dateStr + 'T23:59:59');
  const now = new Date();
  const diff = target - now;
  return Math.ceil(diff / (1000 * 60 * 60 * 24));
}

export function getUrgencyClass(days) {
  if (days < 0) return 'urgency-past';
  if (days <= 3) return 'urgency-critical';
  if (days <= 7) return 'urgency-warning';
  return 'urgency-safe';
}

export function getUrgencyLabel(days) {
  if (days < 0) return 'Past Due';
  if (days === 0) return 'Today!';
  if (days === 1) return 'Tomorrow';
  return `${days} days`;
}

export function searchFilter(items, query, fields) {
  if (!query.trim()) return items;
  const lower = query.toLowerCase();
  return items.filter(item =>
    fields.some(field => {
      const value = item[field];
      return value && String(value).toLowerCase().includes(lower);
    })
  );
}

export function getStatusColor(status) {
  const colors = {
    'Open': 'var(--color-success)',
    'Closed': 'var(--color-danger)',
    'Upcoming': 'var(--color-warning)',
    'Pending': 'var(--color-warning)',
    'Approved': 'var(--color-accent)',
    'Fulfilled': 'var(--color-success)',
    'Denied': 'var(--color-danger)',
    'Checked Out': 'var(--color-warning)',
    'Returned': 'var(--color-success)',
    'Overdue': 'var(--color-danger)',
  };
  return colors[status] || 'var(--color-text-muted)';
}

export function getCategoryColor(category) {
  const colors = {
    'Build': '#f97316',
    'Study': '#3b82f6',
    'Lab': '#10b981',
  };
  return colors[category] || '#8b5cf6';
}

export function getTodayStr() {
  return new Date().toISOString().split('T')[0];
}
