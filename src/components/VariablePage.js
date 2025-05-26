import React from 'react';
import variableData from '../data/variableData';

export default function VariablePage() {
  return (
    <div style={{ padding: '24px', fontFamily: 'Hanna11, sans-serif' }}>
      <h2>⭐ 변수 개념 + 장비 설명</h2>
      <p>8개 변수에 대해 개념 + 대표 장비 + 장비 적용 이유를 3열 8행 형식으로 정리했습니다.</p>

      {variableData.map((item, index) => (
        <div key={index} style={{
          marginBottom: '48px',
          padding: '16px',
          background: '#f9f9f9',
          borderRadius: '12px',
          boxShadow: 'inset 4px 4px 8px #d1d1d1, inset -4px -4px 8px #ffffff'
        }}>
          {/* 🔹 제목 */}
          <h3 style={{ color: item.color }}>{item.label}</h3>

          {/* 🔹 이미지: 변수 + 장비1 + 장비2 */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '12px',
            marginTop: '12px',
            marginBottom: '8px'
          }}>
            <img src={item.variableImage} alt="변수 이미지" style={imageStyle} />
            <img src={item.deviceImage1} alt="장비1" style={imageStyle} />
            <img src={item.deviceImage2} alt="장비2" style={imageStyle} />
          </div>

          {/* 🔹 설명: 개념 + 장비1 설명 + 장비2 설명 */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '12px'
          }}>
            <p style={descStyle}>{item.description}</p>
            <p style={descStyle}><b>📌 장비 1:</b> {item.device1Desc}</p>
            <p style={descStyle}><b>📌 장비 2:</b> {item.device2Desc}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

const imageStyle = {
  width: '100%',
  maxWidth: '260px',
  borderRadius: '8px'
};

const descStyle = {
  fontSize: '14px',
  whiteSpace: 'pre-line',
  color: '#333'
};
