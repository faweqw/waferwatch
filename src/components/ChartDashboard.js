import React, { useEffect, useState, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';
import { Link } from 'react-router-dom';

const allTabs = ['raw', 'rms', 'esp', 'sre', 'gap', 'das', 'csd', 'gpi'];

const explanations = {
  raw: 'RAW: 센서로부터 측정된 전류/진동의 원시 파형입니다.',
  rms: 'RMS: 에너지의 평균 진폭을 나타내며, 진동의 강도를 의미합니다.',
  esp: 'ESP: 주파수 영역에서의 신호 복잡도를 수치화한 값입니다.',
  sre: 'SRE: ESP의 곡률 변화 엔트로피로, 주파수 리듬의 불안정성을 탐지합니다.',
  gap: 'GAP: 고장/정상 간의 SRE 차이를 나타내며, 이상 신호를 정량화합니다.',
  das: 'DAS: GAP의 2차 도함수로 급격한 변화 시점을 파악합니다.',
  csd: 'CSD: SRE 차이의 누적 합이며, 누적된 이상 정도를 표현합니다.',
  gpi: 'GPI: CSD 곡률이 최대가 되는 지점으로, 고장 발생 시점과 관련됩니다.'
};

export default function ChartDashboard() {
  const dataRef = useRef(null);
  const [tab, setTab] = useState('raw');
  const [chartData, setChartData] = useState(null);

  useEffect(() => {
    fetch('/chart_data.json')
      .then(res => res.json())
      .then(json => {
        dataRef.current = json;
        updateChart(json, tab);
      });
  }, []);

  useEffect(() => {
    if (dataRef.current) {
      updateChart(dataRef.current, tab);
    }
  }, [tab]);

  const updateChart = (json, key) => {
    const entry = json[key];
    if (entry) {
      setChartData({
        labels: entry.labels,
        datasets: [{
          label: key.toUpperCase(),
          data: entry.datasets[0].data,
          borderColor: 'rgba(75,192,192,1)',
          fill: false,
          tension: 0.3
        }]
      });
    }
  };

  return (

        <div style={{ background: '#f1f1f4', minHeight: '100vh', padding: '24px', fontFamily: 'sans-serif' }}>


      <h2 style={{ marginBottom: '12px' }}>📊 Waferwatch 그래프 모드</h2>

      {/* 탭 선택 */}
      <div style={{ display: 'flex', gap: '6px', marginBottom: '12px', flexWrap: 'wrap' }}>
        {allTabs.map(key => (
          <button
            key={key}
            onClick={() => setTab(key)}
            style={{
              padding: '6px 12px',
              background: tab === key ? '#007bff' : '#eee',
              color: tab === key ? 'white' : '#333',
              border: 'none',
              borderRadius: '8px',
              boxShadow: '2px 2px 6px rgba(0,0,0,0.1)',
              cursor: 'pointer'
            }}
          >
            {key.toUpperCase()}
          </button>
        ))}
      </div>

      {/* 그래프 */}
      <div style={{ background: '#fff', borderRadius: 8, padding: '16px', height: 400 }}>
        {chartData && (
          <Line
            data={chartData}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              scales: {
                x: { title: { display: true, text: 'Time (s)' } },
                y: {
                  title: { display: true, text: 'Value' },
                  suggestedMin: -6,
                  suggestedMax: 6
                }
              }
            }}
          />
        )}
      </div>
            {/* 📌 설명 카드 */}
      <div style={{
        marginTop: '20px',
        padding: '16px',
        background: '#f9f9f9',
        borderRadius: '12px',
        boxShadow: 'inset 4px 4px 8px #d1d1d1, inset -4px -4px 8px #ffffff',
        fontSize: '14px',
        color: '#333',
        lineHeight: 1.6
      }}>
        <strong>{tab.toUpperCase()}</strong>: {explanations[tab]}
      </div>
    </div>
  );
}
const navButtonStyle = {
  padding: '10px 16px',
  borderRadius: '12px',
  background: '#eee',
  boxShadow: '4px 4px 8px #d1d1d1, -4px -4px 8px #ffffff',
  textDecoration: 'none',
  color: '#333',
  fontWeight: 'bold'
};
