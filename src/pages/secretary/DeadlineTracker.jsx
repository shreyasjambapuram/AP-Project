import { useState, useEffect } from 'react';
import { useDatabase } from '../../hooks/useDatabase';
import { SAMPLE_DEADLINES } from '../../data/sampleData';
import { formatDate, daysUntil, getUrgencyLabel, getUrgencyClass } from '../../utils/helpers';
import { useAuth } from '../../context/AuthContext';

export default function DeadlineTracker() {
  const { isAdmin } = useAuth();
  const { data: deadlines, add, update, remove, loading } = useDatabase('deadlines', SAMPLE_DEADLINES);
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [form, setForm] = useState({ title: '', date: '', category: 'Tournament', notes: '' });

  const sorted = [...deadlines].sort((a, b) => new Date(a.date) - new Date(b.date));

  const handleSave = async (e) => {
    e.preventDefault();
    if (editingId) {
      await update(editingId, form);
    } else {
      await add(form);
    }
    setForm({ title: '', date: '', category: 'Tournament', notes: '' });
    setEditingId(null);
    setShowForm(false);
  };

  const handleEdit = (deadline) => {
    setForm(deadline);
    setEditingId(deadline.id);
    setShowForm(true);
  };

  const handleAddClick = () => {
    setForm({ title: '', date: '', category: 'Tournament', notes: '' });
    setEditingId(null);
    setShowForm(true);
  };

  const handleDelete = async (id) => {
    await remove(id);
  };

  return (
    <div>
      <div className="page-header">
        <h1>Deadline Tracker</h1>
        <p>Never miss a critical administrative date. Color-coded by urgency.</p>
      </div>

      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '1.5rem' }}>
        {isAdmin && <button className="btn btn-primary" onClick={handleAddClick}>+ Add Deadline</button>}
      </div>

      {sorted.length === 0 ? (
        <div className="empty-state glass-card"><div className="empty-icon"></div><p>No deadlines tracked.</p></div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {sorted.map((d, i) => {
            const days = daysUntil(d.date);
            const urgency = getUrgencyClass(days);
            return (
              <div key={d.id} className="glass-card" style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', animationDelay: `${i * 0.05}s`, animation: 'slideUp 0.4s ease backwards' }}>
                <div style={{ minWidth: '80px', textAlign: 'center' }}>
                  <div className={`countdown ${urgency}`}>{days < 0 ? '—' : days}</div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)' }}>{getUrgencyLabel(days)}</div>
                </div>
                <div style={{ width: '3px', height: '40px', borderRadius: '2px', background: days <= 3 ? 'var(--color-danger)' : days <= 7 ? 'var(--color-warning)' : 'var(--color-success)' }} />
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 600, marginBottom: '0.15rem' }}>{d.title}</div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)', display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                    <span>{formatDate(d.date)}</span>
                    <span className={`badge ${d.category === 'Tournament' ? 'badge-accent' : d.category === 'Invoice' ? 'badge-warning' : 'badge-purple'}`}>{d.category}</span>
                  </div>
                  {d.notes && <div style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', marginTop: '0.25rem' }}>{d.notes}</div>}
                </div>
                {isAdmin && (
                  <div style={{ display: 'flex', gap: '0.25rem' }}>
                    <button className="btn btn-icon btn-sm" onClick={() => handleEdit(d)} title="Edit">Edit</button>
                    <button className="btn btn-icon btn-danger btn-sm" onClick={() => handleDelete(d.id)} title="Delete">Del</button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {showForm && (
        <div className="modal-backdrop" onClick={() => setShowForm(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <h2>{editingId ? 'Edit' : 'Add'} Deadline</h2>
            <form onSubmit={handleSave}>
              <div className="form-group"><label className="form-label">Title</label><input className="form-input" required value={form.title} onChange={e => setForm({...form, title: e.target.value})} /></div>
              <div className="form-row">
                <div className="form-group"><label className="form-label">Date</label><input type="date" className="form-input" required value={form.date} onChange={e => setForm({...form, date: e.target.value})} /></div>
                <div className="form-group"><label className="form-label">Category</label><select className="form-select" value={form.category} onChange={e => setForm({...form, category: e.target.value})}><option>Tournament</option><option>Invoice</option><option>Admin</option><option>Social</option></select></div>
              </div>
              <div className="form-group"><label className="form-label">Notes</label><textarea className="form-textarea" value={form.notes} onChange={e => setForm({...form, notes: e.target.value})} /></div>
              <div className="modal-actions"><button type="button" className="btn btn-secondary" onClick={() => setShowForm(false)}>Cancel</button><button type="submit" className="btn btn-primary">Save</button></div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
