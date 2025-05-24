import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';

const dummySignal = () => {
  const x = Array.from({ length: 200 }, (_, i) => i / 10);
  const y = x.map(() => Math.random());
  return { labels: x, datasets: [{ label: 'Dummy Signal', data: y, borderColor: 'blue', tension: 0.2 }] };
};

const explanations = {
  raw: '센서에서 측정한 원신호입니다.',
  rms: 'RMS는 에너지의 크기를 나타냅니다.',
  esp: 'ESP는 주파수 리듬 복잡도를 측정합니다.',
  sre: 'SRE는 주파수 곡률 리듬의 불안정성을 나타냅니다.'
};

const ChartTabs = () => {
  const [tab, setTab] = useState('raw');
  const [data, setData] = useState(dummySignal());

  useEffect(() => {
    const interval = setInterval(() => {
      setData(dummySignal());
    }, 1000);
    return () => clearInterval(interval);
  }, [tab]);

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        {['raw', 'rms', 'esp', 'sre'].map((key) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            style={{
              padding: 8,
              background: tab === key ? '#333' : '#eee',
              color: tab === key ? '#fff' : '#000',
              border: 'none',
              borderRadius: 4
            }}
          >
            {key.toUpperCase()}
          </button>
        ))}
      </div>
      <Line data={data} />
      <p style={{ marginTop: 12 }}>{explanations[tab]}</p>
    </div>
  );
};

export default ChartTabs;
