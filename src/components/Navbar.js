// src/components/Navbar.js
import { Link } from 'react-router-dom';

const navButtonStyle = {
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
      <Link to="/" style={navButtonStyle}>📊 그래프 모드</Link>
      <Link to="/visual" style={navButtonStyle}>🧠 시각화 모드</Link>
    </div>
  );
}
