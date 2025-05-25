// src/components/Navbar.js

import React, { useEffect, useState, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';
import { Link } from 'react-router-dom';

const allTabs = ['raw', 'rms', 'esp', 'sre', 'gap', 'das', 'csd', 'gpi'];

const explanations = {
  raw: "RAW: 센서에서 직접 수집한 원시 신호입니다. 마치 청진기로 들은 심장 소리처럼, 노이즈와 리듬이 가공 없이 담깁니다. 🔑 원신호, 센서, 진동 파형, 노이즈",

  rms: "RMS: 일정 시간 동안의 평균 진폭을 나타냅니다. 이는 맥박의 평균 세기를 보는 것과 같으며, 시스템의 에너지 강도를 파악합니다. 🔑 평균 진폭, 에너지, 강도, 안정성",

  esp: "ESP: 주파수 도메인의 복잡도를 나타냅니다. 오케스트라에 악기가 많아질수록 소리가 복잡해지듯, 주파수 성분이 다양할수록 값이 큽니다. 🔑 스펙트럼 엔트로피, 복잡성, 불확실성, 주파수 분산",

  sre: "SRE: ESP의 곡률 변화량 기반 리듬 복잡성입니다. 심장 박동의 일정한 리듬이 무너지는 순간처럼, 주파수 리듬의 불안정성을 포착합니다. 🔑 리듬 파열, 시간 곡률, 구조적 변화, 엔트로피",

  gap: "GAP: 정상 신호와 고장 신호 간의 리듬 차이입니다. 정상 심장음과 이상 심장음을 비교해 차이를 정량화하는 것과 유사합니다. 🔑 신호 차이, 이상 탐지, 구조적 거리",

  das: "DAS: GAP의 2차 미분값으로, 리듬 변화의 가속도를 나타냅니다. 맥박이 갑자기 뛰는 순간을 감지하는 지표입니다. 🔑 급격한 변화, 민감도, 가속도, 전이점",

  csd: "CSD: GAP의 누적합으로 고장이 점진적으로 축적되는 정도를 나타냅니다. 매일 쌓이는 피로처럼, 고장의 진행성을 시각화합니다. 🔑 누적, 장기 변화, 고장 경향",

  gpi: "GPI: CSD 곡률이 가장 큰 지점, 즉 고장 타이밍의 추정치입니다. 압력이 폭발 직전에 가장 크게 휘어지는 지점과 같습니다. 🔑 예지 타이밍, 최대 곡률, 고장 시작점"
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
