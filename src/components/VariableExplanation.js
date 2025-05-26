// src/components/VariableExplanation.js
import React from 'react';

export default function VariableExplanation() {
  const imageStyle = {
    maxWidth: '300px',
    width: '100%',
    borderRadius: '8px',
    marginBottom: '12px',
    paddingTop: '10px',
    display: 'block'
  };

  const containerStyle = {
    padding: '24px',
    fontFamily: 'Hanna11, sans-serif',
    lineHeight: 1.8
  };

  const variables = [
    {
      name: '🟤 RAW',
      img: '/images/1.png',
      desc: `센서에서 수집한 가공되지 않은 순수 신호입니다.
기계의 떨림, 노이즈, 이상 현상까지 모두 포함된 초기 신호입니다.
이 신호는 아직 분석되지 않아 해석은 어렵지만, 변화가 가장 먼저 나타납니다.
반도체에서는 펌프나 모터의 전류, 진동 신호를 실시간으로 받아오는 단계입니다.
센서 오류나 순간 이상도 포함되어 있어, 그대로 판단하면 과탐 가능성이 있습니다.
따라서 이후의 분석 지표(RMS, ESP 등)의 기초 데이터로 사용됩니다.`
    },
    {
      name: '📈 RMS',
      img: '/images/2.png',
      desc: `신호의 평균 진폭을 수치로 나타낸 것입니다.
얼마나 강하게 진동하거나 전류가 흐르는지를 보여줍니다.
모터의 부하가 커지거나 마찰이 늘어나면 RMS가 상승합니다.
CMP 슬러리 펌프, 회전체 베어링 상태 모니터링에 사용됩니다.
값이 클수록 시스템이 힘들게 동작하고 있을 가능성이 큽니다.`
    },
    {
      name: '🌐 ESP',
      img: '/images/3.jpg',
      desc: `주파수들이 얼마나 다양하게 분포되어 있는지를 나타냅니다.
정상일 때는 특정 주파수대에 집중되지만, 이상이 생기면 여러 주파수가 섞입니다.
값이 클수록 리듬이 복잡하고 불안정하다는 의미입니다.
정속 운전이 필요한 반도체 스테이지나 로봇 구동계에서 활용됩니다.
노이즈나 마운트 불균형 발생 시 ESP가 증가합니다.`
    },
    {
      name: '📉 SRE',
      img: '/images/4.png',
      desc: `시간에 따라 변화하는 ESP 곡선을 분석한 구조 리듬 변화 지표입니다.
일정하던 주파수 패턴이 얼마나 흔들리기 시작했는지를 보여줍니다.
고장 초기에는 리듬 구조가 먼저 무너지므로 조기 감지에 유리합니다.
로딩 암, 리니어 액추에이터의 미세 반복 오차 진단에 적합합니다.
ESP는 복잡함을 보여주고 SRE는 그 변화를 추적합니다.`
    },
    {
      name: '🧭 GAP',
      img: '/images/5.png',
      desc: `현재 신호와 기준 신호 간의 구조 리듬 차이를 수치화한 지표입니다.
값이 클수록 상태가 정상 기준에서 많이 벗어나 있다는 뜻입니다.
Batch 간 비교나 장비 상태 평가에 효과적입니다.
이상 판단의 기준선을 자동화하여 품질 편차 분석에도 활용됩니다.
정량적 비교가 가능한 구조 리듬 거리 지표입니다.`
    },
    {
      name: '⚡ DAS',
      img: '/images/6.jpg',
      desc: `GAP 값이 얼마나 빠르게 변하는지를 나타내는 민감도 높은 지표입니다.
구조 리듬이 갑자기 붕괴되는 시점에 민감하게 반응합니다.
밸브 sticking, 회전체 정렬 이상 등 순간 이상을 포착합니다.
다소 예민하므로 보조 지표와 함께 해석하는 것이 좋습니다.
고장 전이의 가속도 감지에 특화된 지표입니다.`
    },
    {
      name: '📊 CSD',
      img: '/images/7.jpg',
      desc: `GAP의 누적값을 시간축에서 추적한 지표입니다.
작고 반복적인 이상이 장기간 쌓이고 있는지를 보여줍니다.
장시간 운전되는 펌프나 진공 시스템의 누적 피로 평가에 적합합니다.
겉보기에 정상이어도 내부 리듬이 무너지고 있는 경우 CSD가 증가합니다.
장기적인 이상 진단에 유용한 리듬 누적 지표입니다.`
    },
    {
      name: '🚨 GPI',
      img: '/images/8.png',
      desc: `CSD 곡선이 가장 급격히 꺾이는 변곡점을 기준으로 고장 시점을 추정합니다.
고장이 진행되다 전이가 발생하는 시점을 수학적으로 포착합니다.
고장 한 번으로 사고가 날 수 있는 설비에 특히 중요합니다.
실시간 경보나 PM 타이밍 자동화에 활용 가능합니다.
NSI, DAS 등과 조합하여 신뢰도 높은 트리거로 활용됩니다.`
    }
  ];

  return (
    <div style={containerStyle}>
      <h2>⭐ 변수 개념 설명</h2>
      <p>각 변수에 대한 정의, 해석 방식, 응용 예시를 정리했습니다.</p>

      <ul style={{ listStyleType: 'none', paddingLeft: 0 }}>
        {variables.map((v, idx) => (
          <li key={idx} style={{ marginBottom: '32px' }}>
            <img src={v.img} alt={`${v.name} 이미지`} style={imageStyle} />
            <h3>{v.name}</h3>
            {v.desc.split('\n').map((line, i) => (
              <div key={i}>{line}</div>
            ))}
          </li>
        ))}
      </ul>
    </div>
  );
}
