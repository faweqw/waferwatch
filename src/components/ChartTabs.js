import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import { Card, CardContent } from '@/components/ui/card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { motion } from 'framer-motion';
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

const useSimulatedData = (label, interval = 1000) => {
  const [data, setData] = useState({ labels: [], datasets: [] });
  useEffect(() => {
    const update = () => {
      const t = Array.from({ length: 200 }, (_, i) => (i / 10).toFixed(2));
      const y = t.map(() => Math.random());
      setData({
        labels: t,
        datasets: [{
          label,
          data: y,
          borderColor: 'rgba(100, 100, 255, 0.8)',
          fill: false,
          tension: 0.3
        }],
      });
    };
    update();
    const id = setInterval(update, interval);
    return () => clearInterval(id);
  }, [label]);
  return data;
};

const allTabs = ['raw', 'rms', 'esp', 'sre', 'gap', 'das', 'csd', 'gpi'];

export default function WaferwatchDashboard() {
  const [tab, setTab] = useState('raw');
  const chartData = useSimulatedData(tab.toUpperCase());

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-4">Waferwatch 전류/진동 분석 대시보드</h1>
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList className="mb-4 grid grid-cols-4 gap-2">
          {allTabs.map((key) => (
            <TabsTrigger key={key} value={key}>{key.toUpperCase()}</TabsTrigger>
          ))}
        </TabsList>

        {allTabs.map((key) => (
          <TabsContent key={key} value={key}>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <Card className="mb-4">
                <CardContent className="p-4">
                  <Line data={chartData} />
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4 text-base">
                  {explanations[key]}
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
