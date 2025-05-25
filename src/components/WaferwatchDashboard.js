import React, { useEffect, useRef, useState } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Title, Tooltip, Legend);

const variableDescriptions = {
  RAW: "센서에서 측정된 원시 전류 신호입니다.",
  RMS: "신호의 에너지를 나타내는 지표로, 원의 크기로 표현됩니다.",
  ESP: "주파수 영역의 복잡도를 나타내며, 색상 채도로 표현됩니다.",
  SRE: "ESP 곡률의 변화로 리듬의 불안정성을 나타냅니다.",
  GAP: "진폭 간격의 변화로 두 원의 거리로 표현됩니다.",
  DAS: "이동 방향성을 가진 사각형으로, 신호 변화량을 나타냅니다.",
  CSD: "시간에 따른 누적 거리로, 이동 점의 위치로 시각화됩니다.",
  GPI: "이상 징후 경고 지표로, 기준 초과 시 점멸 경고를 발생시킵니다."
};

const variableList = Object.keys(variableDescriptions);

export default function WaferwatchCanvas() {
  const canvasRef = useRef(null);
  const frameRef = useRef(0);
  const dataRef = useRef(null);
  const [selectedVar, setSelectedVar] = useState("RAW");
  const [chartDataMap, setChartDataMap] = useState({});

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    fetch('/chart_data.json')
      .then(res => res.json())
      .then(json => {
        dataRef.current = json;

        // 변수별 chart.js용 그래프 데이터 저장
        const chartData = {};
        for (let key of variableList) {
          if (json[key]) {
            chartData[key] = {
              labels: json[key].labels,
              datasets: [{
                label: key,
                data: json[key].datasets[0].data,
                borderColor: json[key].datasets[0].borderColor,
                tension: 0.3,
                fill: false,
              }]
            };
          }
        }
        setChartDataMap(chartData);
        requestAnimationFrame(draw);
      });

    const getValue = (key, frame) => {
      const d = dataRef.current?.[key.toLowerCase()]?.datasets?.[0]?.data;
      if (!d) return 0;
      return d[frame % d.length] ?? 0;
    };

    const draw = () => {
      const canvas = canvasRef.current;
      if (!canvas || !dataRef.current) {
        requestAnimationFrame(draw);
        return;
      }

      const frame = frameRef.current;
      const ctx = canvas.getContext('2d');
      const cx = canvas.width / 2;
      const cy = canvas.height / 2;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const val = getValue(selectedVar, frame);

      // 예시 시각화: 단순 원 진동
      ctx.beginPath();
      const radius = Math.max(5, 30 + val * 30);
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = "#3498db";
      ctx.fill();

      frameRef.current++;
      requestAnimationFrame(draw);
    };
  }, [selectedVar]);

  return (
    <div style={{ background: '#111', padding: '20px', color: '#fff' }}>
      <h2 className="text-2xl font-bold mb-4">Waferwatch 전류/진동 분석 대시보드</h2>

      <div className="flex flex-wrap gap-2 mb-6">
        {variableList.map((key) => (
          <button
            key={key}
            onClick={() => setSelectedVar(key)}
            className={`px-4 py-2 rounded border ${selectedVar === key ? 'bg-blue-600 text-white' : 'bg-white text-black'}`}
          >
            {key}
          </button>
        ))}
      </div>

      <div className="bg-white text-black rounded p-4 mb-4">
        <h3 className="text-xl font-semibold">{selectedVar}</h3>
        <p className="mt-2 text-sm text-gray-700">{variableDescriptions[selectedVar]}</p>
      </div>

      <div style={{ textAlign: 'center', marginBottom: '20px' }}>
        <canvas ref={canvasRef} width={600} height={300} style={{ background: '#fff' }} />
      </div>

      <div className="bg-white rounded p-4">
        {chartDataMap[selectedVar] ? (
          <Line data={chartDataMap[selectedVar]} />
        ) : (
          <p className="text-gray-500">그래프 데이터를 불러오는 중...</p>
        )}
      </div>
    </div>
  );
}

