//App.js

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import ChartDashboard from './components/ChartDashboard';
import VisualDashboard from './components/VisualDashboard';
import Navbar from './components/Navbar';

export default function App() {
  return (
        <Router>
      <div style={{ background: '#f1f1f4', minHeight: '100vh', fontFamily: 'sans-serif' }}>
        {/* ✅ 상단 고정 내비게이션 바 */}
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          padding: '16px 0',
          borderBottom: '1px solid #ddd',
          position: 'sticky',
          top: 0,
          background: '#f1f1f4',
          zIndex: 10
        }}>
          <Navbar />
        </div>

        {/* ✅ 아래 콘텐츠 영역 */}
        <div style={{ padding: '24px' }}>
          <Routes>
            <Route path="/" element={<ChartDashboard />} />
            <Route path="/visual" element={<VisualDashboard />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}
