import { NavLink } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const sections = [
  {
    title: 'Overview',
    links: [
      { to: '/', icon: '', label: 'Dashboard' },
    ],
  },
  {
    title: 'Secretary',
    links: [
      { to: '/calendar', icon: '', label: 'Calendar' },
      { to: '/signups', icon: '', label: 'SignUp Hub' },
      { to: '/deadlines', icon: '', label: 'Deadlines' },
    ],
  },
  {
    title: 'Study Captain',
    links: [
      { to: '/notes', icon: '', label: 'Note Archive' },
      { to: '/practice-tests', icon: '', label: 'Past Competitions' },
    ],
  },
  {
    title: 'Mentorship',
    links: [
      { to: '/guide', icon: '', label: 'Freshman Guide' },
    ],
  },
];

export default function Sidebar({ isOpen, onClose }) {
  const { isAdmin, logout } = useAuth();

  return (
    <>
      <div className={`sidebar-overlay ${isOpen ? 'open' : ''}`} onClick={onClose} />
      <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="sidebar-logo" style={{ background: 'transparent', padding: 0 }}>
            <img src="/logo.png" alt="SciOly" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
          </div>
          <div>
            <div className="sidebar-title">Bridgeland Highschool SCIOLY</div>
          </div>
        </div>
        <nav className="sidebar-nav" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          <div style={{ flex: 1 }}>
            {sections.map((section) => (
              <div className="nav-section" key={section.title}>
                <div className="nav-section-title">{section.title}</div>
                {section.links.map((link) => (
                  <NavLink
                    key={link.to}
                    to={link.to}
                    end={link.to === '/'}
                    className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                    onClick={onClose}
                  >
                    <span className="nav-icon">{link.icon}</span>
                    {link.label}
                  </NavLink>
                ))}
              </div>
            ))}

            {isAdmin && (
              <div className="nav-section" key="Admin">
                <div className="nav-section-title">Admin</div>
                <NavLink
                  to="/admin/files"
                  className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                  onClick={onClose}
                >
                  <span className="nav-icon"></span>
                  Officer Files
                </NavLink>
              </div>
            )}
          </div>
          
          <div className="nav-section" style={{ marginTop: 'auto', paddingBottom: '1rem' }}>
            <button className="nav-link" onClick={logout} style={{ background: 'transparent', border: 'none', width: '100%', textAlign: 'left', fontFamily: 'inherit' }}>
              <span className="nav-icon"></span> {isAdmin ? 'Admin' : 'Member'} (Logout)
            </button>
          </div>
        </nav>
      </aside>
    </>
  );
}
