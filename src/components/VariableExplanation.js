// src/components/VariableExplanation.js
import React from 'react';

export default function VariableExplanation() {
  const imageStyle = {
    width: '100%',
    maxWidth: '200px',
    height: 'auto',
    objectFit: 'contain',
    borderRadius: '8px',
    marginBottom: '8px'
  };

  const containerStyle = {
    padding: '24px',
    fontFamily: 'Hanna11, sans-serif',
    lineHeight: 1.8
  };

  const variables = [
    {
      name: '🟤 RAW',
      images: ['/images/1.png', '/images/pump.jpeg', '/images/subomotor.jpg'],
      desc: `센서에서 수집한 가공되지 않은 순수 신호입니다.
기계의 떨림, 노이즈, 이상 현상까지 모두 포함된 초기 신호입니다.
이 신호는 아직 분석되지 않아 해석은 어렵지만, 변화가 가장 먼저 나타납니다.
반도체에서는 펌프나 모터의 전류, 진동 신호를 실시간으로 받아오는 단계입니다.
센서 오류나 순간 이상도 포함되어 있어, 그대로 판단하면 과탐 가능성이 있습니다.
따라서 이후의 분석 지표(RMS, ESP 등)의 기초 데이터로 사용됩니다.

펌프:
펌프에서 발생하는 압력 및 유동 변화는 진동 및 전류 형태로 나타납니다.
이러한 신호를 그대로 수집해 초기 고장 징후를 판단할 수 있습니다.
펌프 고장은 신호의 미세한 변화로 먼저 드러나므로 RAW 신호가 핵심입니다.

소형 서보 모터:
모터는 회전 시 일정한 전류 리듬을 유지해야 합니다.
내부 마모나 피로로 인해 리듬이 흐트러지면 RAW에서 가장 먼저 감지됩니다.
특히 실시간 진단 시스템에서 모터의 RAW 신호는 주요 피드백 자료입니다.`
    },
    {
      name: '📈 RMS',
      images: ['/images/2.png', '/images/slurrypump.jpeg', '/images/ballbearing.png'],
      desc: `신호의 평균 진폭을 수치로 나타낸 것입니다.
얼마나 강하게 진동하거나 전류가 흐르는지를 보여줍니다.
모터의 부하가 커지거나 마찰이 늘어나면 RMS가 상승합니다.
값이 클수록 시스템이 힘들게 동작하고 있을 가능성이 큽니다.
단순하지만 직관적이라 이상 탐지의 1차 지표로 널리 활용됩니다.

CMP 슬러리 펌프:
펌프의 압송 성능 저하는 RMS 값 상승으로 감지됩니다.
슬러리 점도 변화나 내부 마모가 반영됩니다.
꾸준한 RMS 모니터링으로 유지보수 주기를 예측할 수 있습니다.

회전체 베어링:
베어링 손상 시 진폭 변화가 즉각적으로 발생합니다.
진동의 평균 세기가 증가하면 마모 또는 이물질 유입 가능성을 시사합니다.
RMS는 베어링 진단에 있어 가장 직관적인 지표입니다.`
    },
    {
      name: '🌐 ESP',
      images: ['/images/3.jpg', '/images/stage.jpg', '/images/drivetrain.png'],
      desc: `주파수들이 얼마나 다양하게 분포되어 있는지를 나타냅니다.
정상일 때는 특정 주파수대에 집중되지만, 이상이 생기면 여러 주파수가 섞입니다.
값이 클수록 리듬이 복잡하고 불안정하다는 의미입니다.
복잡도 기반의 이상 신호 분석에 적합한 주파수 중심 지표입니다.

정밀 스테이지:
스테이지 구동 중 불안정한 주파수 응답은 ESP 증가로 나타납니다.
고정밀 장비일수록 주파수 집중도가 높아야 합니다.
ESP는 성능 열화의 초기 신호를 탐지합니다.

구동 기어계:
기어의 마모나 정렬 불량은 다주파수 성분을 유발합니다.
이로 인해 ESP 값이 점차 증가하게 됩니다.
ESP는 반복성 저하나 진동 비대칭성의 핵심 분석 수단입니다.`
    },
    {
      name: '📉 SRE',
      images: ['/images/4.png', '/images/loadingarm.jpg', '/images/linearactuator.jpg'],
      desc: `시간에 따라 변화하는 ESP 곡선을 분석한 구조 리듬 변화 지표입니다.
일정하던 주파수 패턴이 얼마나 흔들리기 시작했는지를 보여줍니다.
고장 초기에는 리듬 구조가 먼저 무너지므로 조기 감지에 유리합니다.
ESP는 복잡함을 보여주고 SRE는 그 변화를 추적합니다.

로딩 암:
반복적인 로딩 동작에서 일정한 리듬이 유지되어야 합니다.
SRE는 이 리듬이 흔들리는 순간을 즉각적으로 포착합니다.
오차 누적이나 구조 피로 전이를 조기에 감지할 수 있습니다.

리니어 액추에이터:
고속 반복 동작 중 리듬 패턴에 불연속이 발생할 수 있습니다.
SRE는 미세한 불규칙성의 누적을 감지하여 예방 정비를 유도합니다.
장기 사용 장비에 대한 피로 진단에 매우 적합합니다.`
    },
    {
      name: '🧭 GAP',
      images: ['/images/5.png', '/images/loadlock.png', '/images/transfer.jpg'],
      desc: `현재 신호와 기준 신호 간의 구조 리듬 차이를 수치화한 지표입니다.
값이 클수록 상태가 정상 기준에서 많이 벗어나 있다는 뜻입니다.
Batch 간 비교나 장비 상태 평가에 효과적입니다.
상태 비교와 기준화에 적합한 지표입니다.

로드락 챔버:
여러 챔버 간 진동 구조를 GAP으로 비교해 편차를 확인합니다.
동일 조건에서 GAP 값이 다르면 이상 작동 가능성을 시사합니다.
GAP은 장비 간 상대적 이상 진단에 핵심 도구가 됩니다.

이송 로봇:
로봇의 반복 동작 리듬은 기준 신호와 유사해야 합니다.
GAP이 증가하면 마운트 이상이나 정렬 편차를 의심할 수 있습니다.
리듬 비교를 통한 로봇 상태 진단에 활용됩니다.`
    },
    {
      name: '⚡ DAS',
      images: ['/images/6.jpg', '/images/drypump.jpeg', '/images/shaft.png'],
      desc: `GAP 값이 얼마나 빠르게 변하는지를 나타내는 민감도 높은 지표입니다.
갑작스러운 이상 상황에서 민감하게 반응합니다.
가속 기반의 예민한 지표입니다.

드라이 펌프:
밸브 sticking 현상 시 GAP이 급격히 상승합니다.
DAS는 이러한 순간 반응을 실시간으로 포착합니다.
진공 시스템의 돌발 상황 감지에 필수적입니다.

축정렬 시스템:
축 정렬이 틀어지는 순간 진동이 급변합니다.
DAS는 정렬 이상을 빠르게 감지하여 경보를 발생시킬 수 있습니다.
정렬 유지가 중요한 고속 회전체 시스템에 유용합니다.`
    },
    {
      name: '📊 CSD',
      images: ['/images/7.jpg', '/images/cd.png', '/images/chiller.jpg'],
      desc: `GAP의 누적량을 기반으로 장기적인 이상 누적을 분석합니다.
단기 이상보다 피로 축적을 더 잘 반영합니다.
장기 진단에 효과적인 지표입니다.

CD 챔버:
운전 중 발생하는 미세 진동이 장기적으로 누적됩니다.
CSD는 외관상 정상인 상태의 내부 열화를 감지합니다.
고정밀 공정 장비의 내구 평가에 유용합니다.

칠러 유닛:
온도 제어 실패로 발생하는 미세 진동이 누적됩니다.
CSD 상승은 냉각 효율 저하나 펌프 노후화를 암시합니다.
설비 교체 시점 판단에 기여합니다.`
    },
    {
      name: '🚨 GPI',
      images: ['/images/8.png', '/images/robotarm.jpeg', '/images/etchreactor.jpg'],
      desc: `CSD 곡선이 가장 급격히 꺾이는 변곡점을 기준으로 고장 시점을 추정합니다.
고장이 진행되다 전이가 발생하는 시점을 수학적으로 포착합니다.
예지 정비 시스템의 핵심 트리거로서 작동합니다.

로딩 로봇 암:
반복 작업 중 급격한 리듬 붕괴 시점은 GPI로 탐지됩니다.
GPI 피크는 조기 고장 전이의 경고 신호입니다.
정밀 로딩 장비의 사고 방지에 핵심적입니다.

식각 챔버:
고장의 누적이 특정 지점에서 급격히 전이됩니다.
GPI는 이 순간을 수학적으로 포착하여 경보를 보냅니다.
식각 공정 안정성을 위한 마지막 방어선 역할을 합니다.`
    }
  ];

  return (
    <div style={containerStyle}>
      <h2>⭐ 변수 개념 설명</h2>
      <p>각 변수에 대한 정의, 해석 방식, 응용 예시를 정리했습니다.</p>

      <ul style={{ listStyleType: 'none', paddingLeft: 0 }}>
        {variables.map((v, idx) => (
          <li key={idx} style={{ marginBottom: '36px' }}>
            <div style={{ display: 'flex', gap: '16px', marginBottom: '12px' }}>
              {v.images.map((src, i) => (
                <img key={i} src={src} alt={`${v.name} 이미지 ${i + 1}`} style={imageStyle} />
              ))}
            </div>
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
