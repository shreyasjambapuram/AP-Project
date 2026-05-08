import { useState, useEffect } from 'react';
import { useDatabase } from '../../hooks/useDatabase';
import { SAMPLE_SIGNUP_LINKS } from '../../data/sampleData';
import { formatDate, searchFilter, getStatusColor } from '../../utils/helpers';
import { useAuth } from '../../context/AuthContext';

export default function SignUpHub() {
  const { isAdmin } = useAuth();
  const { data: links, add, update, remove, loading } = useDatabase('signup_links', SAMPLE_SIGNUP_LINKS);
  const [filter, setFilter] = useState('All');
  const [search, setSearch] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [form, setForm] = useState({ title: '', category: 'Tournament', url: '', date: '', status: 'Open' });

  const filtered = searchFilter(
    filter === 'All' ? links : links.filter(l => l.category === filter),
    search, ['title']
  );

  const handleSave = async (e) => {
    e.preventDefault();
    if (editingId) {
      await update(editingId, form);
    } else {
      await add(form);
    }
    setForm({ title: '', category: 'Tournament', url: '', date: '', status: 'Open' });
    setEditingId(null);
    setShowForm(false);
  };

  const handleEdit = (link, e) => {
    e.preventDefault();
    e.stopPropagation();
    setForm(link);
    setEditingId(link.id);
    setShowForm(true);
  };

  const handleDelete = async (id, e) => {
    e.preventDefault();
    e.stopPropagation();
    await remove(id);
  };

  const handleAddClick = () => {
    setForm({ title: '', category: 'Tournament', url: '', date: '', status: 'Open' });
    setEditingId(null);
    setShowForm(true);
  };

  return (
    <div>
      <div className="page-header">
        <h1>SignUp Genius Hub</h1>
        <p>All active sign-up links in one place — tournaments, socials, and study hours.</p>
      </div>

      <div className="toolbar">
        <input className="form-input" placeholder="Search links..." value={search} onChange={e => setSearch(e.target.value)} />
        <select className="form-select" value={filter} onChange={e => setFilter(e.target.value)}>
          <option>All</option><option>Tournament</option><option>Social</option><option>Study Hours</option>
        </select>
        {isAdmin && <button className="btn btn-primary" onClick={handleAddClick}>+ Add Link</button>}
      </div>

      {filtered.length === 0 ? (
        <div className="empty-state glass-card"><div className="empty-icon"></div><p>No sign-up links found.</p></div>
      ) : (
        <div className="card-grid">
          {filtered.map((link, i) => (
            <a key={link.id} href={link.url} target="_blank" rel="noreferrer" style={{ textDecoration: 'none' }}>
              <div className="glass-card clickable" style={{ animationDelay: `${i * 0.05}s` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                  <span className={`badge ${link.category === 'Tournament' ? 'badge-accent' : link.category === 'Social' ? 'badge-purple' : 'badge-success'}`}>{link.category}</span>
                  <span className="badge" style={{ background: `${getStatusColor(link.status)}20`, color: getStatusColor(link.status) }}>{link.status}</span>
                </div>
                <h3 style={{ marginBottom: '0.35rem', color: 'var(--color-text)' }}>{link.title}</h3>
                <div style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>{formatDate(link.date)}</div>
                {isAdmin && (
                  <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem' }}>
                    <button className="btn btn-sm" onClick={(e) => handleEdit(link, e)}>Edit</button>
                    <button className="btn btn-sm btn-danger" onClick={(e) => handleDelete(link.id, e)}>Delete</button>
                  </div>
                )}
              </div>
            </a>
          ))}
        </div>
      )}

      {showForm && (
        <div className="modal-backdrop" onClick={() => setShowForm(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <h2>{editingId ? 'Edit' : 'Add'} Sign-Up Link</h2>
            <form onSubmit={handleSave}>
              <div className="form-group"><label className="form-label">Title</label><input className="form-input" required value={form.title} onChange={e => setForm({...form, title: e.target.value})} /></div>
              <div className="form-row">
                <div className="form-group"><label className="form-label">Category</label><select className="form-select" value={form.category} onChange={e => setForm({...form, category: e.target.value})}><option>Tournament</option><option>Social</option><option>Study Hours</option></select></div>
                <div className="form-group"><label className="form-label">Date</label><input type="date" className="form-input" required value={form.date} onChange={e => setForm({...form, date: e.target.value})} /></div>
              </div>
              <div className="form-group"><label className="form-label">URL</label><input className="form-input" required value={form.url} onChange={e => setForm({...form, url: e.target.value})} placeholder="https://signupgenius.com/..." /></div>
              <div className="form-group"><label className="form-label">Status</label><select className="form-select" value={form.status} onChange={e => setForm({...form, status: e.target.value})}><option>Open</option><option>Upcoming</option><option>Closed</option></select></div>
              <div className="modal-actions"><button type="button" className="btn btn-secondary" onClick={() => setShowForm(false)}>Cancel</button><button type="submit" className="btn btn-primary">Save</button></div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
