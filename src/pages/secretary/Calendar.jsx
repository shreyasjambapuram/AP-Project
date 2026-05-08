import { useState, useEffect } from 'react';
import { useDatabase } from '../../hooks/useDatabase';
import { SAMPLE_MEETING_LINKS } from '../../data/sampleData';
import { formatDate } from '../../utils/helpers';
import { useAuth } from '../../context/AuthContext';

export default function Calendar() {
  const { isAdmin } = useAuth();
  const { data: meetings, add, update, remove, loading } = useDatabase('meeting_links', SAMPLE_MEETING_LINKS);
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [form, setForm] = useState({ date: '', title: '', slidesUrl: '', minutesUrl: '' });

  const handleSave = async (e) => {
    e.preventDefault();
    if (editingId) {
      await update(editingId, form);
    } else {
      await add(form);
    }
    setForm({ date: '', title: '', slidesUrl: '', minutesUrl: '' });
    setEditingId(null);
    setShowForm(false);
  };

  const handleEdit = (meeting) => {
    setForm(meeting);
    setEditingId(meeting.id);
    setShowForm(true);
  };

  const handleDelete = async (id) => {
    await remove(id);
  };

  const handleAddClick = () => {
    setForm({ date: '', title: '', slidesUrl: '', minutesUrl: '' });
    setEditingId(null);
    setShowForm(true);
  };

  return (
    <div>
      <div className="page-header">
        <h1>Master Calendar</h1>
        <p>Central view of all meetings, tournaments, and social events.</p>
      </div>

      <div className="glass-card" style={{ marginBottom: '2rem', padding: 0, overflow: 'hidden', borderRadius: 'var(--radius-lg)' }}>
        <iframe
          src="https://calendar.google.com/calendar/embed?src=92aef9df287defac66cbf6ad9e8ace94af5bf0967e0ff0e98873e4c16cddde75%40group.calendar.google.com&ctz=America%2FChicago&bgcolor=%230a0e1a&color=%2300b4d8"
          style={{ width: '100%', height: '500px', border: 'none' }}
          title="Team Calendar"
        />
      </div>

      <div className="section">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <div className="section-title" style={{ margin: 0 }}>Meeting Resources</div>
          {isAdmin && <button className="btn btn-primary" onClick={handleAddClick}>+ Add Link</button>}
        </div>

        {meetings.length === 0 ? (
          <div className="empty-state glass-card"><div className="empty-icon"></div><p>No meeting links yet.</p></div>
        ) : (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Date</th><th>Meeting</th><th>Slides</th><th>Minutes</th>
                  {isAdmin && <th>Actions</th>}
                </tr>
              </thead>
              <tbody>
                {[...meetings].reverse().map(m => (
                  <tr key={m.id}>
                    <td>{formatDate(m.date)}</td>
                    <td style={{ fontWeight: 500, color: 'var(--color-text)' }}>{m.title}</td>
                    <td>{m.slidesUrl ? <a href={m.slidesUrl} target="_blank" rel="noreferrer">Open Slides</a> : '—'}</td>
                    <td>{m.minutesUrl ? <a href={m.minutesUrl} target="_blank" rel="noreferrer">Open Minutes</a> : '—'}</td>
                    {isAdmin && (
                      <td>
                        <div style={{ display: 'flex', gap: '0.5rem' }}>
                          <button className="btn btn-icon btn-sm" onClick={() => handleEdit(m)} title="Edit">Edit</button>
                          <button className="btn btn-icon btn-danger btn-sm" onClick={() => handleDelete(m.id)} title="Delete">Del</button>
                        </div>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {showForm && (
        <div className="modal-backdrop" onClick={() => setShowForm(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <h2>{editingId ? 'Edit' : 'Add'} Meeting Link</h2>
            <form onSubmit={handleSave}>
              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Date</label>
                  <input type="date" className="form-input" required value={form.date} onChange={e => setForm({...form, date: e.target.value})} />
                </div>
                <div className="form-group">
                  <label className="form-label">Meeting Title</label>
                  <input className="form-input" required value={form.title} onChange={e => setForm({...form, title: e.target.value})} />
                </div>
              </div>
              <div className="form-group">
                <label className="form-label">Slides URL</label>
                <input className="form-input" value={form.slidesUrl} onChange={e => setForm({...form, slidesUrl: e.target.value})} placeholder="https://..." />
              </div>
              <div className="form-group">
                <label className="form-label">Minutes URL</label>
                <input className="form-input" value={form.minutesUrl} onChange={e => setForm({...form, minutesUrl: e.target.value})} placeholder="https://..." />
              </div>
              <div className="modal-actions">
                <button type="button" className="btn btn-secondary" onClick={() => setShowForm(false)}>Cancel</button>
                <button type="submit" className="btn btn-primary">Save</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
