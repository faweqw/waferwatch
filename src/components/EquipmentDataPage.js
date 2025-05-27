// src/components/EquipmentDataPage.js
import React from 'react';

export default function EquipmentDataPage() {
  return (
    <div style={{ padding: '24px', fontFamily: 'Hanna11, sans-serif' }}>
      <h2 style={{ fontSize: '1.5rem', marginBottom: '16px' }}>⚙️ 설비 데이터 업로드</h2>

    {/* 반응형 이미지 삽입 */}
<img
  src="/images/18.jpg"
  alt="시뮬레이션 결과 이미지"
  style={{
    width: '30%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>

      <p style={{ marginBottom: '12px', color: '#555' }}>
        CSV 또는 신호 파일을 업로드하세요. 현재는 시각화 없이 파일만 업로드할 수 있습니다.
      </p>
      <input
        type="file"
        accept=".csv"
        onChange={(e) => {
          const file = e.target.files[0];
          if (file) {
            console.log('업로드된 파일:', file.name);
            // TODO: 업로드된 파일 처리 로직 추가
          }
        }}
        style={{
          padding: '8px',
          borderRadius: '6px',
          border: '1px solid #ccc',
          backgroundColor: '#fff',
          cursor: 'pointer'
        }}
      />
    </div>
  );
}
