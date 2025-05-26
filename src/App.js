// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import ChartDashboard from './components/ChartDashboard';
import VisualDashboard from './components/VisualDashboard';
import VariableExplanation from './components/VariableExplanation';

export default function App() {
  return (
    <Router>
      <div style={{ background: '#f1f1f4', minHeight: '100vh', fontFamily: 'sans-serif' }}>
        {/* 상단 내비게이션 */}
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

        {/* 본문 콘텐츠 */}
        <div style={{ padding: '24px' }}>
          <Routes>
            <Route path="/" element={<ChartDashboard />} />
            <Route path="/visual" element={<VisualDashboard />} />
            <Route path="/variables" element={<VariableExplanation />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}
