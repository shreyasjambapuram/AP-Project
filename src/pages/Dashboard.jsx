import { Link } from 'react-router-dom';

export default function Dashboard() {
  return (
    <div>
      <div className="page-header">
        <h1>Bridgeland High School SciOly</h1>
        <p>Welcome to the Science Olympiad Portal. Here is your overview.</p>
      </div>

      <div className="section">
        <div className="section-title">Quick Actions</div>
        <div className="quick-actions">
          <Link to="/calendar" className="quick-action"><span className="qa-icon"></span>Calendar</Link>
          <Link to="/signups" className="quick-action"><span className="qa-icon"></span>SignUp Hub</Link>
          <Link to="/deadlines" className="quick-action"><span className="qa-icon"></span>Deadlines</Link>
          <Link to="/notes" className="quick-action"><span className="qa-icon"></span>Note Archive</Link>
          <Link to="/practice-tests" className="quick-action"><span className="qa-icon"></span>Past Competitions</Link>
          <Link to="/guide" className="quick-action"><span className="qa-icon"></span>Freshman Guide</Link>
        </div>
      </div>
    </div>
  );
}
