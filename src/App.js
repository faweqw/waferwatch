import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import ChartDashboard from './components/ChartDashboard';
import VisualDashboard from './components/VisualDashboard';

export default function App() {
  return (
    <Router>
      <div className="top-nav" style={{ padding: 20 }}>
        <nav style={{ marginBottom: 20 }}>
          <Link to="/" style={{ marginRight: 10 }}>📊 그래프 모드</Link>
          <Link to="/visual">🧠 시각화 모드</Link>
        </nav>
        <Routes>
          <Route path="/" element={<ChartDashboard />} />
          <Route path="/visual" element={<VisualDashboard />} />
        </Routes>
      </div>
    </Router>
  );
}
