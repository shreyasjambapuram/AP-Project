export default function FreshmanGuide() {
  return (
    <div>
      <div className="page-header">
        <h1>Freshman Guide</h1>
        <p>Welcome to Science Olympiad! Here is everything you need to know to get started.</p>
      </div>

      <div className="section">
        <div className="glass-card" style={{ marginBottom: '1.5rem' }}>
          <h2 style={{ color: 'var(--color-accent)', marginBottom: '1rem' }}>What is Science Olympiad?</h2>
          <p style={{ color: 'var(--color-text-secondary)', lineHeight: '1.6' }}>
            Science Olympiad is a premier national science competition where teams of 15 students compete in 23 events spanning various scientific disciplines, including earth science, biology, chemistry, physics, and engineering. 
            Unlike traditional science fairs, SciOly is a team-based competition requiring collaboration, deep studying, and hands-on building.
          </p>
        </div>

        <div className="glass-card" style={{ marginBottom: '1.5rem' }}>
          <h2 style={{ color: 'var(--color-purple)', marginBottom: '1rem' }}>How to Study Effectively</h2>
          <ul style={{ color: 'var(--color-text-secondary)', lineHeight: '1.6', paddingLeft: '1.5rem' }}>
            <li style={{ marginBottom: '0.5rem' }}><strong>Read the Rules:</strong> Always start by thoroughly reading the official rulebook for your events. The rules dictate exactly what can be tested and what materials you can bring.</li>
            <li style={{ marginBottom: '0.5rem' }}><strong>Build a Cheat Sheet:</strong> Many events allow a notes sheet or binder. Do not just print Wikipedia pages! Actively synthesize information into a highly organized, dense cheat sheet.</li>
            <li style={{ marginBottom: '0.5rem' }}><strong>Take Practice Tests:</strong> Use the <em>Practice Tests</em> repository to simulate tournament conditions. Time yourself.</li>
            <li style={{ marginBottom: '0.5rem' }}><strong>Collaborate:</strong> Testing your partner is the best way to learn. Form study groups within your event.</li>
          </ul>
        </div>

        <div className="glass-card">
          <h2 style={{ color: 'var(--color-warning)', marginBottom: '1rem' }}>Competition Etiquette</h2>
          <ul style={{ color: 'var(--color-text-secondary)', lineHeight: '1.6', paddingLeft: '1.5rem' }}>
            <li style={{ marginBottom: '0.5rem' }}><strong>Arrive Early:</strong> For satellite testing or in-person tournaments, punctuality is critical.</li>
            <li style={{ marginBottom: '0.5rem' }}><strong>Bring Materials:</strong> Always bring your own pencils, calculators, and safety goggles (if required). Never assume the proctor will provide them.</li>
            <li style={{ marginBottom: '0.5rem' }}><strong>Stay in the Room:</strong> Do not leave the testing room until dismissed or until the testing block is completely over.</li>
            <li style={{ marginBottom: '0.5rem' }}><strong>Integrity:</strong> Science Olympiad operates on an honor code. Do not communicate with outside individuals during a test.</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
