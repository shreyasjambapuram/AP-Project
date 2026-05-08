const PREFIX = 'scioly_';

export function getData(key) {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function setData(key, value) {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch (e) {
    console.error('Failed to save data:', e);
  }
}

export function addItem(key, item) {
  const list = getData(key) || [];
  const newItem = { ...item, id: Date.now().toString() };
  list.push(newItem);
  setData(key, list);
  return newItem;
}

export function updateItem(key, id, updates) {
  const list = getData(key) || [];
  const idx = list.findIndex(item => item.id === id);
  if (idx !== -1) {
    list[idx] = { ...list[idx], ...updates };
    setData(key, list);
    return list[idx];
  }
  return null;
}

export function removeItem(key, id) {
  const list = getData(key) || [];
  setData(key, list.filter(item => item.id !== id));
}

export function initializeData(key, sampleData) {
  if (!getData(key)) {
    setData(key, sampleData);
  }
  return getData(key);
}
