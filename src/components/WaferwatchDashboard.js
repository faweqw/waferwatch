import React, { useEffect, useRef, useState } from 'react';
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
  gpi: 'GPI는 CSD의 곡률이 최대인 지점으로, 고장 발생 시점을 추정합니다.',
};

const allTabs = ['raw', 'rms', 'esp', 'sre', 'gap', 'das', 'csd', 'gpi'];

export default function WaferwatchFullDashboard() {
  const canvasRef = useRef(null);
  const frameRef = useRef(0);
  const dataRef = useRef(null);
  const [tab, setTab] = useState('raw');
  const [showChart, setShowChart] = useState(true);
  const [chartData, setChartData] = useState(null);

  useEffect(() => {
    fetch('/chart_data.json')
      .then(res => res.json())
      .then(json => {
        dataRef.current = json;
        updateChart(json);
        requestAnimationFrame(draw);
      });
  }, []);

  const updateChart = (json) => {
    const entry = json[tab];
    if (entry) {
      const labels = entry.labels;
      const data = entry.datasets[0].data;
      setChartData({
        labels,
        datasets: [{
          label: tab.toUpperCase(),
          data,
          borderColor: 'rgba(75,192,192,1)',
          fill: false,
          tension: 0.3
        }]
      });
    }
  };

useEffect(() => {
  const draw = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');

    if (!dataRef.current || !canvas || showChart) {
      requestAnimationFrame(draw);
      return;
    }

    const frame = frameRef.current;
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const getSafeValue = (key) => {
      const arr = dataRef.current?.[key]?.datasets?.[0]?.data;
      return arr ? arr[frame % arr.length] ?? 0 : 0;
    };

    const raw = getSafeValue("raw");
    const rms = getSafeValue("rms");
    const esp = getSafeValue("esp");
    const sre = getSafeValue("sre");
    const gap = getSafeValue("gap");
    const das = getSafeValue("das");
    const csd = getSafeValue("csd");
    const gpi = getSafeValue("gpi");

    // raw
    ctx.beginPath();
    ctx.moveTo(0, cy);
    for (let x = 0; x < canvas.width; x++) {
      const y = cy + Math.sin(x * 0.05 + frame * 0.1) * 5 + raw * 2;
      ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#ccc";
    ctx.stroke();

    // rms
    ctx.beginPath();
    ctx.arc(cx - 150, cy, Math.max(1, 30 + rms * 40), 0, Math.PI * 2);
    ctx.fillStyle = "#3498db";
    ctx.fill();

    // esp
    ctx.beginPath();
    ctx.arc(cx + 150, cy, 30, 0, Math.PI * 2);
    ctx.fillStyle = `hsl(200, ${Math.min(Math.max(esp * 100, 0), 100)}%, 50%)`;
    ctx.fill();

    // sre
    const offset = Math.sin(frame * 0.3 + sre) * 10;
    ctx.fillStyle = "#f1c40f";
    ctx.fillRect(cx - 30 + offset, cy - 100, 40, 40);

    // gap
    const dGap = gap * 100;
    ctx.beginPath();
    ctx.arc(cx - dGap / 2, cy + 100, 20, 0, Math.PI * 2);
    ctx.arc(cx + dGap / 2, cy + 100, 20, 0, Math.PI * 2);
    ctx.fillStyle = "#e67e22";
    ctx.fill();

    // das
    ctx.fillStyle = "#2ecc71";
    ctx.fillRect(cx + das * 30, cy - 150, 30, 30);

    // csd
    ctx.beginPath();
    ctx.arc((csd * 20 + canvas.width) % canvas.width, 30, 6, 0, Math.PI * 2);
    ctx.fillStyle = "#9b59b6";
    ctx.fill();

    // gpi
    if (gpi > 1.5 && Math.floor(frame / 10) % 2 === 0) {
      ctx.beginPath();
      ctx.arc(cx, 50, 10, 0, Math.PI * 2);
      ctx.fillStyle = "red";
      ctx.fill();
    }

    frameRef.current++;
    requestAnimationFrame(draw);
  };

  requestAnimationFrame(draw);
}, [showChart]);


  return (
    <div style={{ background: '#111', padding: '20px', color: '#fff', fontFamily: 'sans-serif' }}>
      <h2 style={{ marginBottom: '12px' }}>Waferwatch 통합 대시보드</h2>

      {/* 탭 버튼 */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '12px' }}>
        {allTabs.map(key => (
          <button
            key={key}
            onClick={() => setTab(key)}
            style={{
              padding: '6px 12px',
              background: tab === key ? '#007bff' : '#333',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}>
            {key.toUpperCase()}
          </button>
        ))}
        <button
          onClick={() => setShowChart(!showChart)}
          style={{
            marginLeft: 'auto',
            background: '#555',
            padding: '6px 12px',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            color: 'white'
          }}>
          {showChart ? '🧠 시각화 보기' : '📊 그래프 보기'}
        </button>
      </div>

      {/* 메인 콘텐츠 */}
      {showChart ? (
        <div style={{ background: '#fff', borderRadius: 8, padding: '16px' }}>
          {chartData && <Line data={chartData} />}
        </div>
      ) : (
        <canvas ref={canvasRef} width={800} height={400} style={{ background: '#fff', borderRadius: 8 }} />
      )}

      {/* 설명 카드 */}
      <div style={{
        marginTop: '16px',
        padding: '12px',
        background: '#222',
        borderRadius: '8px',
        fontSize: '16px',
        lineHeight: 1.5
      }}>
        <strong>{tab.toUpperCase()}</strong>: {explanations[tab]}
      </div>
    </div>
  );
}
