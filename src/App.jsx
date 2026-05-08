import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Calendar from './pages/secretary/Calendar';
import SignUpHub from './pages/secretary/SignUpHub';
import DeadlineTracker from './pages/secretary/DeadlineTracker';
import NoteArchive from './pages/study-captain/NoteArchive';
import PracticeTests from './pages/study-captain/PracticeTests';
import FreshmanGuide from './pages/mentorship/FreshmanGuide';
import Login from './pages/Login';
import OfficerFiles from './pages/admin/OfficerFiles';
import { AuthProvider, useAuth } from './context/AuthContext';

function ProtectedRoute({ children }) {
  const { isMember } = useAuth();
  if (!isMember) {
    return <Navigate to="/login" replace />;
  }
  return children;
}

function AdminRoute({ children }) {
  const { isAdmin } = useAuth();
  if (!isAdmin) {
    return <Navigate to="/" replace />;
  }
  return children;
}

export default function App() {
  return (
    <AuthProvider>
      <HashRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          
          <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
            <Route index element={<Dashboard />} />
            
            {/* Secretary Portal */}
            <Route path="calendar" element={<Calendar />} />
            <Route path="signups" element={<SignUpHub />} />
            <Route path="deadlines" element={<DeadlineTracker />} />
            
            {/* Study Captain Portal */}
            <Route path="notes" element={<NoteArchive />} />
            <Route path="practice-tests" element={<PracticeTests />} />

            {/* Mentorship Portal */}
            <Route path="guide" element={<FreshmanGuide />} />

            {/* Admin Portal */}
            <Route path="admin/files" element={<AdminRoute><OfficerFiles /></AdminRoute>} />
            
            {/* Placeholders for upcoming portals */}
            <Route path="*" element={<div className="empty-state"><h2>Under Construction</h2><p>This portal section is being built right now!</p></div>} />
          </Route>
        </Routes>
      </HashRouter>
    </AuthProvider>
  );
}
