import React, { useEffect, useRef } from 'react';

export default function VisualDashboard() {
  const canvasRef = useRef(null);
  const dataRef = useRef(null);
  const frameRef = useRef(0);

  const draw = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!dataRef.current || !canvas || !ctx) return;

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

    // raw waveform
    ctx.beginPath();
    ctx.moveTo(0, cy);
    for (let x = 0; x < canvas.width; x++) {
      const y = cy + Math.sin(x * 0.05 + frame * 0.1) * 5 + raw * 2;
      ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#ccc";
    ctx.stroke();

    // 기타 시각화 요소 동일 (위 코드 참고)
        // rms: 왼쪽 원
    ctx.beginPath();
    ctx.arc(cx - 220, cy, Math.max(1, 30 + rms * 40), 0, Math.PI * 2);
    ctx.fillStyle = "#3498db";
    ctx.fill();

    // esp: 오른쪽 원
    ctx.beginPath();
    ctx.arc(cx + 220, cy, 30, 0, Math.PI * 2);
    ctx.fillStyle = `hsl(200, ${Math.min(Math.max(esp * 100, 0), 100)}%, 50%)`;
    ctx.fill();

    // sre: 진동하는 네모
    const offset = Math.sin(frame * 0.3 + sre) * 10;
    ctx.fillStyle = "#f1c40f";
    ctx.fillRect(cx - 30 + offset, cy - 100, 40, 40);

    // gap: 두 원 간격
    const dGap = gap * 100;
    ctx.beginPath();
    ctx.arc(cx - dGap / 2, cy + 100, 20, 0, Math.PI * 2);
    ctx.arc(cx + dGap / 2, cy + 100, 20, 0, Math.PI * 2);
    ctx.fillStyle = "#e67e22";
    ctx.fill();

    // das: 위 사각형 x 이동
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


    frameRef.current++;
    setTimeout(() => requestAnimationFrame(draw), 330);
  };

  useEffect(() => {
    fetch('/chart_data.json')
      .then(res => res.json())
      .then(json => {
        dataRef.current = json;
        requestAnimationFrame(draw);
      });
  }, []);

  return (
    <div>
      <h2>🧠 Waferwatch 시각화 모드</h2>
      <canvas
        ref={canvasRef}
        width={800}
        height={400}
        style={{ background: '#fff', borderRadius: 8 }}
      />
    </div>
  );
}
