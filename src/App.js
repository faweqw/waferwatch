import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import ChartDashboard from './components/ChartDashboard';
import VisualDashboard from './components/VisualDashboard';
import Navbar from './components/Navbar';

export default function App() {
  return (
    <Router>
      <Navbar/>
      <div className="top-nav" style={{ padding: 20 }}>
        <Routes>
          <Route path="/" element={<ChartDashboard />} />
          <Route path="/visual" element={<VisualDashboard />} />
        </Routes>
      </div>
    </Router>
  );
}
