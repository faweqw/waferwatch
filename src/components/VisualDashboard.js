// src/components/VisualDashboard.js

import React, { useEffect, useRef } from 'react';

export default function VisualDashboard() {
  const canvasRef = useRef(null);
  const dataRef = useRef(null);
  const frameRef = useRef(0);
  const lastTimeRef = useRef(performance.now());
  const speedFactor = 0.25; // 🔧 1.0 = 기본 속도, 0.25 = 4배 느림

  const draw = (now) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!dataRef.current || !canvas || !ctx) return;

    // 시간 기반 프레임 증가
    const deltaTime = now - lastTimeRef.current;
    lastTimeRef.current = now;
    frameRef.current += speedFactor * deltaTime / 16.67; // 16.67 ≈ 60fps 기준
    const frame = Math.floor(frameRef.current);

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

    // === 시각화 요소 ===

    // raw waveform
    ctx.beginPath();
    ctx.moveTo(0, cy);
    for (let x = 0; x < canvas.width; x++) {
      const y = cy + Math.sin(x * 0.05 + frame * 0.1) * 5 + raw * 2;
      ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#ccc";
    ctx.stroke();

    // rms: 왼쪽 원
    ctx.beginPath();
    ctx.arc(cx - 220, cy, Math.max(1, 30 + rms * 40), 0, Math.PI * 2);
    ctx.fillStyle = "#3498db";
    ctx.fill();

    // esp: 오른쪽 원 색상
    ctx.beginPath();
    ctx.arc(cx + 220, cy, 30, 0, Math.PI * 2);
    ctx.fillStyle = `hsl(200, ${Math.min(Math.max(esp * 100, 0), 100)}%, 50%)`;
    ctx.fill();

    // sre: 진동하는 사각형
    const offset = Math.sin(frame * 0.3 + sre) * 10;
    ctx.fillStyle = "#f1c40f";
    ctx.fillRect(cx - 30 + offset, cy - 100, 40, 40);

    // gap: 두 원 사이 거리
    const dGap = gap * 100;
    ctx.beginPath();
    ctx.arc(cx - dGap / 2, cy + 100, 20, 0, Math.PI * 2);
    ctx.arc(cx + dGap / 2, cy + 100, 20, 0, Math.PI * 2);
    ctx.fillStyle = "#e67e22";
    ctx.fill();

    // das: 위쪽 사각형 이동
    ctx.fillStyle = "#2ecc71";
    ctx.fillRect(cx + das * 30, cy - 150, 30, 30);

    // csd: 좌우로 움직이는 점
    ctx.beginPath();
    ctx.arc((csd * 20 + canvas.width) % canvas.width, 30, 6, 0, Math.PI * 2);
    ctx.fillStyle = "#9b59b6";
    ctx.fill();

    // gpi: 깜빡이는 빨간 점
    if (gpi > 1.5 && Math.floor(frame / 10) % 2 === 0) {
      ctx.beginPath();
      ctx.arc(cx, 50, 10, 0, Math.PI * 2);
      ctx.fillStyle = "red";
      ctx.fill();
    }

    // 다음 프레임
    requestAnimationFrame(draw);
  };

  useEffect(() => {
    fetch('/chart_data.json')
      .then(res => res.json())
      .then(json => {
        dataRef.current = json;
        requestAnimationFrame(draw); // 첫 프레임 시작
      });
  }, []);

  return (
  <div>
    <div style={{ padding: '20px', background: '#f4f4f4', fontFamily: 'sans-serif' }}>
      <h2 style={{ marginBottom: '12px' }}>🧠 Waferwatch 시각화 모드</h2>
      <canvas
        ref={canvasRef}
        width={800}
        height={400}
        style={{
          background: '#fff',
          borderRadius: '8px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
        }}
      />

    </div>
      <div style={{
        marginTop: '24px',
        padding: '16px',
        background: '#f9f9f9',
        borderRadius: '12px',
        boxShadow: 'inset 4px 4px 8px #d1d1d1, inset -4px -4px 8px #ffffff',
        fontSize: '14px',
        lineHeight: 1.6,
        color: '#333'
      }}>
        <strong>🎨 변수 설명:</strong>
  <ul style={{ marginTop: '10px', paddingLeft: '20px' }}>
    <li><span style={{ color: '#666' }}><strong>RAW</strong></span>: 원신호 (센서 전류/진동 파형)</li>
    <li><span style={{ color: '#1565c0' }}><strong>RMS</strong></span>: 진폭 평균 (파란 원 크기)</li>
    <li><span style={{ color: '#0277bd' }}><strong>ESP</strong></span>: 주파수 복잡도 (오른쪽 원 색상)</li>
    <li><span style={{ color: '#f39c12' }}><strong>SRE</strong></span>: 리듬 불안정성 (진동하는 네모)</li>
    <li><span style={{ color: '#ef6c00' }}><strong>GAP</strong></span>: SRE 차이 (아래 두 원 거리)</li>
    <li><span style={{ color: '#388e3c' }}><strong>DAS</strong></span>: GAP 가속도 (위 사각형 이동)</li>
    <li><span style={{ color: '#8e24aa' }}><strong>CSD</strong></span>: 누적 GAP (움직이는 보라 점)</li>
    <li><span style={{ color: '#c62828' }}><strong>GPI</strong></span>: 고장 추정 지점 (깜빡이는 점)</li>
  </ul>
      </div>


  </div>


  );
}
