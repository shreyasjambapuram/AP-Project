import { useState, useEffect } from 'react';
import { useDatabase } from '../../hooks/useDatabase';
import { SAMPLE_PRACTICE_TESTS } from '../../data/sampleData';
import { searchFilter } from '../../utils/helpers';
import { useAuth } from '../../context/AuthContext';

export default function PracticeTests() {
  const { isAdmin } = useAuth();
  const { data: tests, add, update, remove, loading } = useDatabase('practice_tests_3', SAMPLE_PRACTICE_TESTS);
  const [search, setSearch] = useState('');
  const [filterDiff, setFilterDiff] = useState('All');
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [form, setForm] = useState({ title: '', difficulty: 'Regional', year: '2025', url: '' });

  const filtered = searchFilter(
    filterDiff === 'All' ? tests : tests.filter(t => t.difficulty === filterDiff),
    search, ['title', 'year']
  );

  const handleSave = async (e) => {
    e.preventDefault();
    if (editingId) {
      await update(editingId, form);
    } else {
      await add(form);
    }
    setForm({ title: '', difficulty: 'Regional', year: '2025', url: '' });
    setEditingId(null);
    setShowForm(false);
  };

  const handleEdit = (test) => {
    setForm(test);
    setEditingId(test.id);
    setShowForm(true);
  };

  const handleDelete = async (id) => {
    await remove(id);
  };

  const handleAddClick = () => {
    setForm({ title: '', difficulty: 'Regional', year: '2025', url: '' });
    setEditingId(null);
    setShowForm(true);
  };

  return (
    <div>
      <div className="page-header">
        <h1>Past Competitions</h1>
        <p>A repository of past competitions and their test packets.</p>
      </div>

      <div className="toolbar">
        <input className="form-input" placeholder="Search competitions..." value={search} onChange={e => setSearch(e.target.value)} />
        <select className="form-select" value={filterDiff} onChange={e => setFilterDiff(e.target.value)}>
          <option>All</option><option>Regional</option><option>State</option><option>National</option><option>Invitational</option>
        </select>
        {isAdmin && <button className="btn btn-primary" onClick={handleAddClick}>+ Add Competition</button>}
      </div>

      {filtered.length === 0 ? (
        <div className="empty-state glass-card"><div className="empty-icon"></div><p>No tests found.</p></div>
      ) : (
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Year</th><th>Competition Name</th><th>Difficulty</th><th>Link</th>
                {isAdmin && <th>Actions</th>}
              </tr>
            </thead>
            <tbody>
              {filtered.map(t => (
                <tr key={t.id}>
                  <td>{t.year}</td>
                  <td style={{ fontWeight: 500, color: 'var(--color-text)' }}>{t.title}</td>
                  <td><span className={`badge ${t.difficulty === 'State' ? 'badge-purple' : t.difficulty === 'National' ? 'badge-danger' : 'badge-accent'}`}>{t.difficulty}</span></td>
                  <td><a href={t.url} target="_blank" rel="noreferrer">Open Tests</a></td>
                  {isAdmin && (
                    <td>
                      <div style={{ display: 'flex', gap: '0.5rem' }}>
                        <button className="btn btn-icon btn-sm" onClick={() => handleEdit(t)} title="Edit">Edit</button>
                        <button className="btn btn-icon btn-danger btn-sm" onClick={() => handleDelete(t.id)} title="Delete">Del</button>
                      </div>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {showForm && (
        <div className="modal-backdrop" onClick={() => setShowForm(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <h2>{editingId ? 'Edit' : 'Add'} Competition</h2>
            <form onSubmit={handleSave}>
              <div className="form-group"><label className="form-label">Competition Name</label><input className="form-input" required value={form.title} onChange={e => setForm({...form, title: e.target.value})} /></div>
              <div className="form-row">
                <div className="form-group"><label className="form-label">Year</label><input type="number" className="form-input" required value={form.year} onChange={e => setForm({...form, year: e.target.value})} /></div>
              </div>
              <div className="form-row">
                <div className="form-group"><label className="form-label">Difficulty</label><select className="form-select" value={form.difficulty} onChange={e => setForm({...form, difficulty: e.target.value})}><option>Regional</option><option>State</option><option>National</option><option>Invitational</option></select></div>
                <div className="form-group"><label className="form-label">URL</label><input className="form-input" required value={form.url} onChange={e => setForm({...form, url: e.target.value})} placeholder="https://..." /></div>
              </div>
              <div className="modal-actions"><button type="button" className="btn btn-secondary" onClick={() => setShowForm(false)}>Cancel</button><button type="submit" className="btn btn-primary">Save</button></div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
