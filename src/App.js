// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import ChartDashboard from './components/ChartDashboard';
import VisualDashboard from './components/VisualDashboard';
import VariableExplanation from './components/VariableExplanation';
import EquipmentDataPage from './components/EquipmentDataPage';
import SimulationAnalysisPage from './components/SimulationAnalysisPage'; // 상단에 추가
import PaperAnalysisPage from './components/PaperAnalysisPage'; // 추가
import IndustryApplicationPage from './components/IndustryApplicationPage'; // 상단에 import 추가



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
            <Route path="/data" element={<EquipmentDataPage />} /> 
            <Route path="/simulation" element={<SimulationAnalysisPage />} />
            <Route path="/paper" element={<PaperAnalysisPage />} /> {/* 추가 */}
            <Route path="/industry" element={<IndustryApplicationPage />} /> {/* ✅ 추가 */}
          </Routes>
        </div>
      </div>
    </Router>
  );
}
