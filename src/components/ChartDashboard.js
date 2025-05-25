import React, { useEffect, useState, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';

const allTabs = ['raw', 'rms', 'esp', 'sre', 'gap', 'das', 'csd', 'gpi'];

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
    <div>
      <h2>📊 Waferwatch 그래프 모드</h2>
      <div style={{ display: 'flex', gap: '6px', marginBottom: '12px' }}>
        {allTabs.map(key => (
          <button key={key} onClick={() => setTab(key)}>{key.toUpperCase()}</button>
        ))}
      </div>
      <div style={{ background: '#fff', borderRadius: 8, padding: '16px', height: 400 }}>
        {chartData && (
          <Line
            data={chartData}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              scales: {
                x: { title: { display: true, text: 'Time (s)' } },
                y: { title: { display: true, text: 'Value' }, suggestedMin: -1, suggestedMax: 1 }
              }
            }}
          />
        )}
      </div>
    </div>
  );
}
