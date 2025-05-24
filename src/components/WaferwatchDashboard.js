import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';

const explanations = {
  raw: '센서에서 측정한 전류/진동 원신호입니다.',
  rms: 'RMS는 시간 창 기준 에너지 평균으로 진동 강도를 나타냅니다.',
  esp: 'ESP는 주파수 영역의 복잡도를 나타냅니다.',
  sre: 'SRE는 ESP 곡률 변화의 엔트로피로, 리듬 불안정성을 포착합니다.',
  gap: 'GAP는 고장 신호 SRE와 정상 신호 SRE의 차이를 나타냅니다.',
  das: 'DAS는 GAP의 곡률(가속도)로 급격한 변화점을 탐지합니다.',
  csd: 'CSD는 두 SRE 간의 누적 차이를 나타내는 지표입니다.',
  gpi: 'GPI는 CSD의 곡률이 최대인 지점으로, 고장 발생 시점을 추정합니다.'
};

const allTabs = ['raw', 'rms', 'esp', 'sre', 'gap', 'das', 'csd', 'gpi'];

export default function WaferwatchDashboard() {
  const [tab, setTab] = useState('raw');
  const [chartData, setChartData] = useState(null);

  useEffect(() => {
    fetch('/chart_data.json')
      .then(res => res.json())
      .then(data => setChartData(data))
      .catch(err => console.error("Failed to load chart data:", err));
  }, []);

  if (!chartData) {
    return <div style={{ padding: '20px' }}>로딩 중... (Loading chart data)</div>;
  }

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '16px' }}>
        Waferwatch 전류/진동 분석 대시보드
      </h1>

      {/* 탭 버튼 */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '20px' }}>
        {allTabs.map((key) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            style={{
              padding: '8px 12px',
              border: '1px solid #ccc',
              background: tab === key ? '#007bff' : '#f9f9f9',
              color: tab === key ? 'white' : 'black',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            {key.toUpperCase()}
          </button>
        ))}
      </div>

      {/* 차트 카드 */}
      <div style={{ border: '1px solid #ddd', padding: '16px', borderRadius: '8px', marginBottom: '12px' }}>
        <Line data={chartData[tab]} />
      </div>

      {/* 설명 카드 */}
      <div style={{ border: '1px solid #eee', padding: '16px', borderRadius: '8px', background: '#fafafa' }}>
        {explanations[tab]}
      </div>
    </div>
  );
}
