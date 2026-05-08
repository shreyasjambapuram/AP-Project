import { useState, useEffect } from 'react';
import { useDatabase } from '../../hooks/useDatabase';
import { SAMPLE_NOTES } from '../../data/sampleData';
import { searchFilter } from '../../utils/helpers';
import { useAuth } from '../../context/AuthContext';

export default function NoteArchive() {
  const { isAdmin } = useAuth();
  const { data: notes, add, update, remove, loading } = useDatabase('notes', SAMPLE_NOTES);
  const [search, setSearch] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [form, setForm] = useState({ title: '', event: '', season: '2025-2026', author: '', description: '', url: '' });

  const filtered = searchFilter(notes, search, ['title', 'event', 'author']);

  const handleSave = async (e) => {
    e.preventDefault();
    if (editingId) {
      await update(editingId, form);
    } else {
      await add(form);
    }
    setForm({ title: '', event: '', season: '2025-2026', author: '', description: '', url: '' });
    setEditingId(null);
    setShowForm(false);
  };

  const handleEdit = (note, e) => {
    e.preventDefault();
    e.stopPropagation();
    setForm(note);
    setEditingId(note.id);
    setShowForm(true);
  };

  const handleDelete = async (id, e) => {
    e.preventDefault();
    e.stopPropagation();
    await remove(id);
  };

  const handleAddClick = () => {
    setForm({ title: '', event: '', season: '2025-2026', author: '', description: '', url: '' });
    setEditingId(null);
    setShowForm(true);
  };

  return (
    <div>
      <div className="page-header">
        <h1>Note Archive</h1>
        <p>A searchable repository for member-uploaded study notes, cheat sheets, and binders.</p>
      </div>

      <div className="toolbar">
        <input className="form-input" placeholder="Search notes, events, authors..." value={search} onChange={e => setSearch(e.target.value)} />
        <button className="btn btn-primary" onClick={handleAddClick}>+ Add Note</button>
      </div>

      {filtered.length === 0 ? (
        <div className="empty-state glass-card"><div className="empty-icon"></div><p>No notes found.</p></div>
      ) : (
        <div className="card-grid">
          {filtered.map((note, i) => (
            <a key={note.id} href={note.url} target="_blank" rel="noreferrer" style={{ textDecoration: 'none' }}>
              <div className="glass-card clickable" style={{ animationDelay: `${i * 0.05}s` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                  <span className="badge badge-accent">{note.event}</span>
                  <span className="badge badge-purple">{note.season}</span>
                </div>
                <h3 style={{ marginBottom: '0.35rem', color: 'var(--color-text)' }}>{note.title}</h3>
                <div style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginBottom: '0.75rem' }}>{note.description}</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>By {note.author}</div>
                
                {isAdmin && (
                  <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem' }}>
                    <button className="btn btn-sm" onClick={(e) => handleEdit(note, e)}>Edit</button>
                    <button className="btn btn-sm btn-danger" onClick={(e) => handleDelete(note.id, e)}>Delete</button>
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
            <h2>{editingId ? 'Edit' : 'Add'} Note</h2>
            <form onSubmit={handleSave}>
              <div className="form-group"><label className="form-label">Title</label><input className="form-input" required value={form.title} onChange={e => setForm({...form, title: e.target.value})} /></div>
              <div className="form-row">
                <div className="form-group"><label className="form-label">Event</label><input className="form-input" required value={form.event} onChange={e => setForm({...form, event: e.target.value})} /></div>
                <div className="form-group"><label className="form-label">Season</label><input className="form-input" required value={form.season} onChange={e => setForm({...form, season: e.target.value})} /></div>
              </div>
              <div className="form-group"><label className="form-label">Author</label><input className="form-input" required value={form.author} onChange={e => setForm({...form, author: e.target.value})} /></div>
              <div className="form-group"><label className="form-label">Description</label><textarea className="form-textarea" required value={form.description} onChange={e => setForm({...form, description: e.target.value})} /></div>
              <div className="form-group"><label className="form-label">URL (Link to doc/pdf)</label><input className="form-input" required value={form.url} onChange={e => setForm({...form, url: e.target.value})} placeholder="https://..." /></div>
              <div className="modal-actions"><button type="button" className="btn btn-secondary" onClick={() => setShowForm(false)}>Cancel</button><button type="submit" className="btn btn-primary">Save</button></div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
