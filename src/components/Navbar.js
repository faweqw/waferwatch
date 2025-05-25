// src/components/Navbar.js
import { Link } from 'react-router-dom';

export default function Navbar() {
  return (
    <div style={{ display: 'flex', gap: '12px', marginBottom: '20px' }}>
      <Link to="/" style={navButtonStyle}>📊 그래프 모드</Link>
      <Link to="/visual" style={navButtonStyle}>🧠 시각화 모드</Link>
    </div>
  );
}
