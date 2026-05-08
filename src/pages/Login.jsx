import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Login() {
  const [role, setRole] = useState('member'); // 'member' or 'admin'
  const [sNumber, setSNumber] = useState('');
  const [passcode, setPasscode] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    const result = await login(sNumber, passcode, role);
    
    setLoading(false);
    
    if (result.success) {
      navigate('/');
    } else {
      setError(result.error);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '2rem',
      background: 'var(--color-bg)',
      color: 'var(--color-text)'
    }}>
      <div className="glass-card" style={{ maxWidth: '400px', width: '100%', padding: '2.5rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1rem' }}>
            <img src="/logo.png" alt="SciOly Logo" style={{ width: '80px', height: '80px', objectFit: 'contain' }} />
          </div>
          <h1 style={{ margin: 0, fontSize: '1.5rem', color: 'var(--color-text)' }}>Bridgeland Highschool SCIOLY</h1>
          <p style={{ color: 'var(--color-text-secondary)', marginTop: '0.5rem' }}>Authorized Personnel Only</p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '2rem', background: 'rgba(255,255,255,0.05)', padding: '0.25rem', borderRadius: 'var(--radius-md)' }}>
          <button 
            type="button"
            style={{ 
              flex: 1, 
              padding: '0.5rem', 
              background: role === 'member' ? 'var(--color-purple)' : 'transparent', 
              color: role === 'member' ? '#fff' : 'var(--color-text-secondary)',
              border: 'none', 
              borderRadius: 'var(--radius-sm)', 
              cursor: 'pointer',
              fontWeight: 500,
              transition: 'all 0.2s'
            }}
            onClick={() => { setRole('member'); setError(''); }}
          >
            Member
          </button>
          <button 
            type="button"
            style={{ 
              flex: 1, 
              padding: '0.5rem', 
              background: role === 'admin' ? 'var(--color-accent)' : 'transparent', 
              color: role === 'admin' ? '#fff' : 'var(--color-text-secondary)',
              border: 'none', 
              borderRadius: 'var(--radius-sm)', 
              cursor: 'pointer',
              fontWeight: 500,
              transition: 'all 0.2s'
            }}
            onClick={() => { setRole('admin'); setError(''); }}
          >
            Admin
          </button>
        </div>

        <form onSubmit={handleLogin}>
          <div className="form-group">
            <label className="form-label">Student ID (S-Number)</label>
            <input 
              type="text" 
              className="form-input" 
              placeholder="e.g. S123456" 
              value={sNumber}
              onChange={(e) => setSNumber(e.target.value)}
              required
            />
          </div>

          {role === 'admin' && (
            <div className="form-group" style={{ animation: 'fadeIn 0.2s ease-out' }}>
              <label className="form-label">Admin Passcode</label>
              <input 
                type="password" 
                className="form-input" 
                placeholder="Enter passcode" 
                value={passcode}
                onChange={(e) => setPasscode(e.target.value)}
                required
              />
            </div>
          )}

          {error && (
            <div style={{ 
              padding: '0.75rem', 
              background: 'rgba(239, 68, 68, 0.1)', 
              borderLeft: '4px solid var(--color-danger)', 
              color: 'var(--color-danger)', 
              fontSize: '0.9rem',
              marginBottom: '1.5rem',
              borderRadius: '0 4px 4px 0'
            }}>
              {error}
            </div>
          )}

          <button type="submit" className="btn btn-primary" style={{ width: '100%', padding: '0.75rem', fontSize: '1rem' }} disabled={loading}>
            {loading ? 'Verifying...' : 'Access Portal'}
          </button>
        </form>
      </div>
    </div>
  );
}
