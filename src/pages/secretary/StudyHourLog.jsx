import { useState, useEffect } from 'react';
import { initializeData, addItem, getData } from '../../utils/localStorage';
import { SAMPLE_MEMBERS, SAMPLE_STUDY_HOURS, ELIGIBILITY_THRESHOLD } from '../../data/sampleData';
import { EVENTS } from '../../data/events';
import { formatDate, searchFilter, getTodayStr } from '../../utils/helpers';
import { useAuth } from '../../context/AuthContext';

export default function StudyHourLog() {
  const { isAdmin } = useAuth();
  const [members, setMembers] = useState([]);
  const [hours, setHours] = useState([]);
  const [search, setSearch] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({ memberName: '', date: getTodayStr(), duration: '', event: '', notes: '' });

  useEffect(() => {
    setMembers(initializeData('members', SAMPLE_MEMBERS));
    setHours(initializeData('study_hours', SAMPLE_STUDY_HOURS));
  }, []);

  const memberHours = members.map(m => {
    const mHours = hours.filter(h => h.memberName === m.name);
    const total = m.hours + mHours.reduce((s, h) => s + Number(h.duration), 0);
    const lastSession = mHours.length ? mHours.sort((a, b) => b.date.localeCompare(a.date))[0].date : null;
    return { ...m, totalHours: total, lastSession, eligible: total >= ELIGIBILITY_THRESHOLD };
  });

  const filtered = searchFilter(memberHours, search, ['name']);

  const handleAdd = (e) => {
    e.preventDefault();
    const item = addItem('study_hours', { ...form, memberId: Date.now().toString() });
    setHours(prev => [...prev, item]);
    setForm({ memberName: '', date: getTodayStr(), duration: '', event: '', notes: '' });
    setShowForm(false);
  };

  return (
    <div>
      <div className="page-header">
        <h1>Study Hour Log</h1>
        <p>Track study hours for tournament eligibility. Minimum {ELIGIBILITY_THRESHOLD} hours required.</p>
      </div>

      <div className="toolbar">
        <input className="form-input" placeholder="Search members..." value={search} onChange={e => setSearch(e.target.value)} />
        {isAdmin && <button className="btn btn-primary" onClick={() => setShowForm(true)}>+ Log Hours</button>}
      </div>

      <div className="table-container">
        <table>
          <thead><tr><th>Member</th><th>Grade</th><th>Total Hours</th><th>Progress</th><th>Last Session</th><th>Status</th></tr></thead>
          <tbody>
            {filtered.map(m => (
              <tr key={m.id}>
                <td style={{ fontWeight: 500, color: 'var(--color-text)' }}>{m.name}</td>
                <td>{m.grade}</td>
                <td style={{ fontFamily: 'var(--font-heading)', fontWeight: 600 }}>{m.totalHours}h</td>
                <td style={{ minWidth: '120px' }}>
                  <div className="progress-bar">
                    <div className="progress-fill" style={{ width: `${Math.min(100, (m.totalHours / ELIGIBILITY_THRESHOLD) * 100)}%`, background: m.eligible ? 'var(--color-success)' : 'var(--color-accent)' }} />
                  </div>
                </td>
                <td>{m.lastSession ? formatDate(m.lastSession) : '—'}</td>
                <td><span className={`badge ${m.eligible ? 'badge-success' : 'badge-warning'}`}>{m.eligible ? 'Eligible' : 'In Progress'}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {showForm && (
        <div className="modal-backdrop" onClick={() => setShowForm(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <h2>Log Study Hours</h2>
            <form onSubmit={handleAdd}>
              <div className="form-group"><label className="form-label">Member Name</label>
                <select className="form-select" required value={form.memberName} onChange={e => setForm({...form, memberName: e.target.value})}>
                  <option value="">Select member...</option>
                  {members.map(m => <option key={m.id} value={m.name}>{m.name}</option>)}
                </select>
              </div>
              <div className="form-row">
                <div className="form-group"><label className="form-label">Date</label><input type="date" className="form-input" required value={form.date} onChange={e => setForm({...form, date: e.target.value})} /></div>
                <div className="form-group"><label className="form-label">Duration (hours)</label><input type="number" step="0.5" min="0.5" className="form-input" required value={form.duration} onChange={e => setForm({...form, duration: e.target.value})} /></div>
              </div>
              <div className="form-group"><label className="form-label">Event Studied</label>
                <select className="form-select" value={form.event} onChange={e => setForm({...form, event: e.target.value})}>
                  <option value="">Select event...</option>
                  {EVENTS.map(ev => <option key={ev.id} value={ev.id}>{ev.icon} {ev.name}</option>)}
                </select>
              </div>
              <div className="form-group"><label className="form-label">Notes</label><textarea className="form-textarea" value={form.notes} onChange={e => setForm({...form, notes: e.target.value})} /></div>
              <div className="modal-actions"><button type="button" className="btn btn-secondary" onClick={() => setShowForm(false)}>Cancel</button><button type="submit" className="btn btn-primary">Log Hours</button></div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
