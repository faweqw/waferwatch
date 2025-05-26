// src/components/Navbar.js
import React from 'react';
import { Link } from 'react-router-dom';

const navButtonStyle = {
  fontFamily: 'Hanna11, sans-serif',
  padding: '10px 16px',
  borderRadius: '12px',
  background: '#eee',
  boxShadow: '4px 4px 8px #d1d1d1, -4px -4px 8px #ffffff',
  textDecoration: 'none',
  color: '#333',
  fontWeight: 'bold'
};

export default function Navbar() {
  return (
    <div style={{ display: 'flex', gap: '12px', marginBottom: '20px' }}>
      <Link to="/variables" style={navButtonStyle}>⭐ 변수 개념 설명</Link>
      <Link to="/" style={navButtonStyle}>📊 그래프 모드</Link>
      <Link to="/visual" style={navButtonStyle}>🧠 시각화 모드</Link>
      <Link to="/" style={navButtonStyle}>⚙️ 설비 데이터 추가</Link>
    
    </div>
  );
}
