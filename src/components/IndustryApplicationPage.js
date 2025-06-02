// src/components/IndustryApplicationPage.js
import React from 'react';

export default function IndustryApplicationPage() {
  const tocStyle = {
    textDecoration: 'none',
    background: '#f5f5f5',
    padding: '8px 12px',
    borderRadius: '8px',
    fontSize: '0.95rem',
    color: '#333',
    fontWeight: 'bold',
    boxShadow: '2px 2px 4px #ccc',
    transition: 'background 0.3s',
  };

  return (
    <div style={{ padding: '24px', fontFamily: 'Hanna11, sans-serif' }}>
      <h2 style={{ fontSize: '1.5rem', marginBottom: '16px' }}>📦 산업 적용 사례</h2>

{/* 목차 */}
<div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginBottom: '16px' }}>
  {/* 리듬 분석 패러다임 */}
  <a href="#legacy" style={tocStyle}>📊 전통 분석 방식</a>
  <a href="#ai" style={tocStyle}>🤖 AI 분석 방식</a>
  <a href="#rhythm" style={tocStyle}>🎵 구조 리듬 분석</a>
  <a href="#commercialization" style={tocStyle}>😎 SRE 상용화 과제</a>
  <a href="#sre-nsi" style={tocStyle}>🚧 민감성 문제 대응</a>
  <a href="#realtime" style={tocStyle}>⚙️ 실시간성 확보</a>
  <a href="#connectivity" style={tocStyle}>🔌 설비 연동 구조</a>
  <a href="#labeling" style={tocStyle}>🏷️ 라벨링 문제 해결</a>
  <a href="#interpretability" style={tocStyle}>🧠 해석 가능성 개선</a>
</div>

<h2 style={{ fontSize: '1.5rem', marginBottom: '16px' }}>🌐 5대 반도체 장비사 적용 사례</h2>

<div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginBottom: '16px' }}>
  {/* 장비별 분석 */}
  <a href="#amat-endura-report" style={tocStyle}>🛠️ AMAT Endura 분석</a>
  <a href="#asml-euv-scanner-report" style={tocStyle}>🔬 ASML EUV 개요</a>
  <a href="#asml-euv-diagnostics-report" style={tocStyle}>⚠️ ASML EUV 고장 분석</a>
  <a href="#tel-trias-overview" style={tocStyle}>🏢 TEL Trias SPA 개요</a>
  <a href="#tel-trias-analysis" style={tocStyle}>🧪 TEL Trias SPA 고장 분석</a>
    <a href="#lam-altus-analysis" style={tocStyle}>🔧 Lam Research ALTUS CVD 분석</a>
  <a href="#lam-altus-failure-analysis" style={tocStyle}>⚠️ Lam Research ALTUS 고장 및 리듬 분석</a>
  <a href="#kla-surfscan-mechanism" style={tocStyle}>🔍 KLA Surfscan 동작 메커니즘</a>
  <a href="#kla-surfscan-sre-gpi" style={tocStyle}>📈 KLA Surfscan SRE/GPI 분석</a>
</div>

<h2 style={{ fontSize: '1.5rem', marginBottom: '16px' }}>📈 SRE/GPI 경제성 분석</h2>

<div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginBottom: '16px' }}>
  <a href="#sre-gpi-economic-scenarios" style={tocStyle}>💰 경제적 가치: 시나리오 비교</a>
  <a href="#samsung-sk-line-application" style={tocStyle}>🏭 삼성전자 & SK하이닉스 라인 구조</a>
  <a href="#realistic-application-scenario" style={tocStyle}>⚙️ 현실적 적용 시나리오</a>
  <a href="#economic-estimation" style={tocStyle}>📊 경제적 가치 추정 (삼성 DRAM 기준)</a>
  <a href="#risk-and-response" style={tocStyle}>🧠 도입 리스크 및 대응 전략</a>
  <a href="#conclusion" style={tocStyle}>✅ 결론 요약</a>
  <a href="#sk-hynix-application" style={tocStyle}>🏢 SK하이닉스 전용 적용 전략</a>
</div>

    {/* 반응형 이미지 삽입 */}
<img
  src="/images/c1.png"
  alt="시뮬레이션 결과 이미지"
  style={{
    width: '30%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>

<h3 id="legacy" style={{ marginTop: '32px', fontWeight: 'bold' }}>📊 전통적 분석 방식 (RMS, FFT 등)</h3>
<p style={{ lineHeight: '1.8' }}>
  전통적 고장 진단 방식은 산업계에서 수십 년간 사용되어 온 안정적인 분석 프레임워크입니다. 대표적으로 RMS(Root Mean Square), 
  FFT(Fast Fourier Transform), Peak Ratio, Crest Factor, Z-score 분석 등이 있으며, 주로 진동, 전류, 압력과 같은 물리 센서로부터 
  수집된 데이터를 기반으로 고장을 감지합니다. 이러한 방식은 신호의 ‘크기’나 ‘주파수 분포’에 집중하기 때문에 구현이 비교적 쉽고, 
  실시간 분석과 시스템 통합이 매우 용이하다는 장점이 있습니다. 특히 RMS는 에너지 레벨을 정량화하여 급격한 출력 변화를 탐지하고, 
  FFT는 회전체의 불균형이나 고정밀 진동 문제에서 특정 고장 주파수를 분석하는 데 사용됩니다.

  그러나 이러한 방식은 몇 가지 핵심적인 한계를 가지고 있습니다. 첫째, 대부분의 전통 분석 기법은 <b>선형 시스템 가정</b>에 기반합니다. 
  실제 산업 설비는 온도, 하중, 공정조건 등에 따라 비선형적 거동을 하기 때문에, 작고 점진적인 이상 상태는 감지되지 않고 넘어가는 경우가 많습니다. 
  둘째, 기준 임계값을 사람이 정해야 하는 경우가 대부분이며, 이 임계값은 환경 변화에 민감하게 반응합니다. 
  공정 조건이 조금만 달라져도 오탐지나 미탐지 가능성이 급격히 높아지며, 기준선 Drift 문제가 발생하기도 합니다.

  가장 중요한 문제는 이러한 방식이 대부분 ‘결과 기반(Response-based)’ 방식이라는 점입니다. 즉, 고장이 이미 발생하거나 
  임계값을 넘은 이후에야 경고를 발생시키는 구조이므로, 조기 감지나 예지 보전(Predictive Maintenance)에는 취약합니다. 
  이로 인해 생산 수율 저하, 장비 가동 중단 등의 리스크를 미연에 방지하기 어렵습니다. 실제로 많은 현장에서는 경고 발생 직후의 
  고장 대응만으로 시스템을 운영하고 있으며, 고장을 피하는 것이 아니라 '대응'에만 그치는 한계를 보입니다.
</p>

    {/* 반응형 이미지 삽입 */}
<img
  src="/images/c2.jpg"
  alt="시뮬레이션 결과 이미지"
  style={{
    width: '30%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>


<h3 id="ai" style={{ marginTop: '32px', fontWeight: 'bold' }}>🤖 AI 기반 분석 방식 (CNN, LSTM 등)</h3>
<p style={{ lineHeight: '1.8' }}>
  AI 기반 고장 진단은 최근 산업계에서 급부상한 방법론으로, 딥러닝 알고리즘을 활용해 설비의 상태를 분류하거나 예측하는 데 사용됩니다. 
  대표적으로 CNN(Convolutional Neural Network)은 시간 또는 주파수 영역의 특징을 자동으로 추출하고, 
  LSTM(Long Short-Term Memory)은 시계열 데이터 내의 장기적인 의존성을 학습할 수 있습니다. 최근에는 Transformer 모델이 
  고성능 연산 장비와 함께 도입되면서, 산업 신호 분석에도 점차 확장되고 있습니다. 이러한 방식은 <b>비선형성과 다변량 구조</b>, 
  그리고 <b>비정형 데이터의 패턴 인식</b>에 탁월한 성능을 보여주며, 전통 방식으로는 감지하지 못했던 미세 이상 상태나 복합 고장을 분류할 수 있습니다.

  그러나 AI 기반 방식은 구조적으로 <b>설명력이 부족(Black-box)</b> 하다는 문제가 큽니다. 설비 엔지니어는 왜 고장으로 판별되었는지를 
  판단하기 어렵고, 예측 결과에 대한 해석도 직관적이지 않습니다. 이는 현장의 실질적인 의사결정과 연동되지 못하는 경우가 많으며, 
  고장이 나도 ‘왜 그런지’를 설명할 수 없기 때문에 신뢰성 확보에 어려움이 따릅니다. 또한, AI 모델은 <b>학습 데이터의 질과 양에 매우 민감</b>합니다. 
  이상 상태에 대한 라벨이 부족하거나 노이즈가 포함된 경우, 모델은 쉽게 과적합(overfitting)되거나 성능 저하를 보입니다.

  더불어, 설비가 변경되거나 새로운 환경이 도입될 경우 <b>재학습 비용이 매우 높습니다.</b> 기존 모델은 유연하지 않고 
  특정 조건에만 최적화되어 있기 때문에, 설비 변경 시마다 전체 시스템을 튜닝하거나 다시 훈련해야 하는 부담이 큽니다. 
  이로 인해 실제로 AI 모델이 정밀하게 만들어져도 <b>운영 유지보수 측면에서 비용과 시간이 많이 드는 비효율적인 결과</b>가 발생하기도 합니다.
</p>

    {/* 반응형 이미지 삽입 */}
<img
  src="/images/c3.png"
  alt="시뮬레이션 결과 이미지"
  style={{
    width: '30%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>


<h3 id="rhythm" style={{ marginTop: '32px', fontWeight: 'bold' }}>🎵 구조 리듬 기반 분석 (SRE, GPI 등)</h3>
<p style={{ lineHeight: '1.8' }}>
  구조 리듬 기반 분석은 시간축에서 신호의 구조적 일관성과 리듬성을 중심으로 이상 여부를 감지하는 최신 분석 방법입니다. 
  이 방식은 고전적 크기 분석(RMS)이나 주파수 분석(FFT)과 달리, 신호의 '모양'과 '형태적 흐름'을 중심으로 해석합니다. 
  SRE(Spectral Rhythm Entropy)는 신호의 시간적 반복성 및 복잡성을 정보 이론적으로 정량화하며, GPI(Geometric Phase Index)는 
  곡률의 시간 변화율을 기반으로 한 리듬 붕괴의 정도를 측정합니다. 이는 고장이 발생하기 수 시간 전, 시스템 내 <b>구조적 안정성이 무너지는 순간</b>을 
  조기에 포착할 수 있는 매우 강력한 지표입니다.

  가장 큰 장점은 이 방식이 <b>물리 기반 해석을 가능하게 하며, 결과에 대한 설명력이 뛰어나다는 점입니다.</b> 엔지니어는 GPI가 상승하는 이유를 
  곡률 변화나 리듬성 붕괴와 연결시켜 설비의 구조적 문제로 해석할 수 있으며, 이를 바탕으로 구체적인 예방 조치를 설계할 수 있습니다. 
  또한, 기존 방식이 감지하지 못하는 정렬 불량, 편심, 미세 진동 불균형, 반복성 저하 등을 매우 민감하게 포착할 수 있습니다. 
  특히 회전체 설비, 로딩 암, 드라이펌프 등에서 그 효과가 뛰어납니다.

  그러나 이 방식 역시 단점이 존재합니다. 구조 리듬 기반 분석은 일반적인 통계 기법보다 계산량이 많고, 
  고주파 노이즈가 포함된 신호에서는 곡률 계산이 왜곡될 수 있습니다. 또한, 설비마다 기준 리듬을 설정해주어야 하며, 
  이를 구축하는 데 초기 시간이 소요됩니다. 실시간 적용을 위해서는 GPI 계산의 최적화가 필수적이며, 해석 기준도 전문가 수준의 이해를 요구합니다. 
  그럼에도 불구하고, 이 방식은 기존 전통 방식과 AI 방식의 단점을 보완하며, 설명 가능하고 조기 감지가 가능한 <b>중간지대 기술로서 매우 유망</b>한 접근으로 평가받고 있습니다.
</p>


<h3 id="commercialization" style={{ marginTop: '32px', fontWeight: 'bold' }}>😎 SRE 기반 기술의 상용화 장애물과 해결 전략</h3>
<p style={{ lineHeight: '1.8' }}>
  구조 리듬 기반 고장 진단 기법인 SRE(Spectral Rhythm Entropy)는 이론적으로 뛰어난 조기 감지 성능과 해석 가능성을 보유하고 있습니다. 
  하지만 산업 현장에 적용하기 위해서는 다음과 같은 5가지 기술적 장애물을 반드시 해결해야 합니다.
  <br/><br/>
  <strong>🚧 고차 미분 기반 수치 민감성</strong><br/>
  SRE는 ESP 곡선의 2차 미분(곡률)을 이용하므로, 센서 노이즈나 이상치에도 민감하게 반응하여 잘못된 경고(false peak)가 발생할 수 있습니다. 
  이를 해결하기 위해 Savitzky-Golay 필터 기반의 곡률 안정화, NSI(Noise Stability Index) 보조지표 도입, 또는 주기성 기반 변화율 중심의 
  SRE-lite 모델로 단순화하는 방식이 효과적입니다.
  <br/><br/>
  <strong>⚙️ 실시간성 확보의 어려움</strong><br/>
  실시간 분석을 위해 SRE는 Sliding Window 내에서 연산량이 많은 KDE와 Shannon 엔트로피 계산을 반복 수행해야 합니다. 
  따라서 실시간 제어 시스템(PLC, FPGA 등)에서 지연이 발생할 수 있습니다. 이를 해결하기 위해 KDE 대신 히스토그램 근사 방식이나 LUT 기반 엔트로피 계산을 도입하고, 
  dSRE/dt 기반의 변화율 분석 및 이벤트 중심 슬라이딩 구조(Event-driven Window)를 적용해 연산량을 줄일 수 있습니다.
  <br/><br/>
  <strong>🔌 산업 설비 연동성 부족</strong><br/>
  많은 공장 설비들은 FFT나 RMS 기반 모니터링에 최적화되어 있어, 곡률 기반 지표를 실시간으로 수신하거나 처리할 수 있는 구조가 부족합니다. 
  이를 보완하기 위해 REST API 기반 SRE 연산 결과를 OPC UA, MQTT 등의 프로토콜로 변환해 전달하는 중간 모듈이 필요하며, 
  PLC 시스템과 병렬 운영 가능한 임베디드 형태의 SRE-lite 모듈도 설계되어야 합니다.
  <br/><br/>
  <strong>🏷️ 정확한 고장 라벨 부족</strong><br/>
  대부분의 산업 데이터셋은 고장 발생 시점을 명확히 포함하지 않아 성능 검증이 어렵습니다. 이를 보완하기 위해 SRE 기반 약지도(weak label)를 생성하고, 
  SRE_peak이 실제 고장보다 몇 초 앞섰는지를 기준화하여 평가하는 방식을 사용할 수 있습니다. 
  또한 라벨이 없는 환경에서는 시뮬레이션 기반 고장 패턴을 생성해 학습 데이터로 사용할 수 있습니다.
  <br/><br/>
  <strong>🧠 해석 가능성 부족</strong><br/>
  SRE, 곡률, 엔트로피, KDE 등의 개념은 일반 엔지니어에게는 직관적이지 않기 때문에, 현장 적용 시 거부감이 발생할 수 있습니다. 
  이를 해결하기 위해 M-C-K 진동 모델 기반의 물리 해석을 함께 제공하고, 
  SRE/NSI/ESP 값을 병렬 시각화한 후 간단한 자연어 설명 메시지(예: "72시간 내 베어링 교체 권고")를 제공하는 Rule-based 진단 보조 시스템이 필요합니다.
  <br/><br/>
  <strong>📌 총평</strong><br/>
  위의 문제점들은 단순한 기술적 한계를 넘어, 실제 적용성과 운용 가능성을 좌우하는 핵심 요소입니다. 
  그러나 SRE는 경량화와 보조지표 도입을 통해 실시간 진단 시스템으로 진화할 수 있으며, 
  적절한 해석 프레임워크와 시각화 도구를 갖추면 산업계 수용성을 획기적으로 높일 수 있습니다. 
  이는 기존 AI 기반 이상 탐지와 전통 RMS 기반 방식 사이의 중간지대 기술로서, 설명 가능하면서도 실시간 경고가 가능한 고장 진단 플랫폼으로 발전할 가능성이 매우 큽니다.
</p>

<h3 id="sre-nsi" style={{ marginTop: '32px', fontWeight: 'bold' }}>🚧 수치 민감성 문제 해결 – SRE-lite + NSI 병렬 구조</h3>

<p style={{ lineHeight: '1.8' }}>
  ESP의 곡률 계산은 2차 미분을 포함하므로 고주파 노이즈나 이상치에 민감하여 false peak가 자주 발생합니다. 
  이를 완화하기 위해 ESP 시계열을 Savitzky-Golay 필터로 평활화한 후, 곡률을 계산하고, 
  NSI(Noise Spread Index)와 병렬로 조건을 만족할 경우에만 이벤트를 발생시킵니다.
</p>

<h4 style={{ fontWeight: 'bold', marginTop: '24px' }}>✅ Python 코드 (SRE-lite + NSI 병렬 구조)</h4>

<pre style={{ background: '#f5f5f5', padding: '16px', borderRadius: '8px', overflowX: 'auto' }}>
<code>
{`
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# 예시 신호: 노이즈 포함된 ESP 시계열
np.random.seed(42)
t = np.linspace(0, 10, 2000)
esp_signal = np.sin(2 * np.pi * 0.5 * t) + 0.3 * np.random.randn(len(t))

# 1. Savitzky-Golay 필터로 평활화
esp_smoothed = savgol_filter(esp_signal, window_length=51, polyorder=3)

# 2. 곡률 계산 (2차 도함수 기반)
d1 = np.gradient(esp_smoothed)
d2 = np.gradient(d1)
kappa = d2 / (1e-6 + (1 + d1**2)**1.5)

# 3. NSI 계산 (최근 N포인트 구간)
def compute_nsi(signal, window_size=200):
    segment = signal[-window_size:]
    return np.std(segment) / (np.mean(np.abs(segment)) + 1e-6)

nsi = compute_nsi(esp_smoothed)

# 4. 이벤트 트리거 조건
kappa_threshold = 0.8
nsi_threshold = 0.25
trigger = np.any(np.abs(kappa) > kappa_threshold) and nsi > nsi_threshold

print("🚨 리듬 붕괴 경고 발생 여부:", trigger)

# 5. 시각화
plt.figure(figsize=(12, 5))
plt.plot(t, esp_signal, label="원시 ESP", alpha=0.5)
plt.plot(t, esp_smoothed, label="평활화 ESP", linewidth=2)
plt.title("🎵 ESP 시계열 및 Savitzky-Golay 평활화 결과")
plt.xlabel("시간 (초)")
plt.ylabel("진폭")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
`}
</code>
</pre>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  위 코드는 노이즈가 포함된 ESP 신호를 곡률 기반으로 분석할 때 발생할 수 있는 오탐 문제를 완화하기 위한 전략입니다.
  NSI는 잡음이 확산되었는지를 정량화하는 지표로, 구조적 리듬 붕괴가 일어났는지 확인하는 보조 수단이 됩니다.
  <strong>SRE-lite + NSI 병렬 조건을 만족할 때만 경고를 발생시키므로</strong> 신뢰도 있는 이상 탐지가 가능해집니다.
</p>

<h3 id="realtime" style={{ marginTop: '32px', fontWeight: 'bold' }}>⚙️ 실시간성 확보 문제 – 경량화 구조 (KDE 근사, 변화율 기반)</h3>

<p style={{ lineHeight: '1.8' }}>
  SRE는 고차 미분, 커널 밀도 추정(KDE), Shannon Entropy 등을 Sliding Window 내에서 반복 계산해야 하므로
  PLC, FPGA 등 실시간 제어 시스템에 적용하기엔 연산 부담이 큽니다.
  이를 해결하기 위해 KDE 대신 단순 히스토그램 기반 근사, 또는 변화율 기반 경량화 구조(dSRE/dt)를 사용하여 처리 속도를 개선할 수 있습니다.
</p>

<h4 style={{ fontWeight: 'bold', marginTop: '24px' }}>✅ Python 코드 (히스토그램 기반 Entropy 근사 + dSRE/dt)</h4>

<pre style={{ background: '#f5f5f5', padding: '16px', borderRadius: '8px', overflowX: 'auto' }}>
<code>
{`
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

# 샘플 데이터 생성 (진폭 변동이 있는 신호)
np.random.seed(0)
signal = np.sin(np.linspace(0, 20*np.pi, 2000)) + 0.3*np.random.randn(2000)

# Sliding Window 기반 Entropy 계산
def compute_sre_histogram(signal, window_size=200, step=20, bins=32):
    sre_vals = []
    for i in range(0, len(signal) - window_size, step):
        segment = signal[i:i+window_size]
        hist, _ = np.histogram(segment, bins=bins, density=True)
        p = hist / np.sum(hist)
        p = p[p > 0]
        sre = -np.sum(p * np.log2(p))
        sre_vals.append(sre)
    return np.array(sre_vals)

sre_values = compute_sre_histogram(signal)
sre_diff = np.diff(sre_values)  # 변화율 (dSRE/dt)

# 시각화
plt.figure(figsize=(12, 5))
plt.plot(sre_values, label="SRE (Entropy)")
plt.plot(sre_diff, label="dSRE/dt", linestyle='--')
plt.title("📉 SRE 값과 변화율 기반 경량화 구조")
plt.xlabel("윈도우 인덱스")
plt.ylabel("SRE / 변화량")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
`}
</code>
</pre>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  위 코드는 Sliding Entropy를 <strong>히스토그램 기반으로 근사</strong>하여 실시간 계산 부담을 줄이고,
  변화율(dSRE/dt)을 통해 구조 리듬의 급격한 붕괴를 빠르게 감지할 수 있도록 구성됩니다.
  특히 임베디드 시스템에서는 <strong>Shannon Entropy 계산을 LUT 또는 정수 기반 연산으로 치환</strong>함으로써 연산 최적화가 가능합니다.
</p>

<h3 id="connectivity" style={{ marginTop: '32px', fontWeight: 'bold' }}>🔌 설비 연동성 문제 – OPC UA / MQTT 통합 구조</h3>

<p style={{ lineHeight: '1.8' }}>
  대부분의 산업 장비는 RMS나 FFT 기반 진단에 최적화되어 있어, 곡률 기반 지표(SRE/GPI)를 연동하기 어렵습니다.
  이를 해결하기 위해선 OPC UA나 MQTT 프로토콜을 통해 구조 리듬 지표를 외부 시스템으로 전송하고, 
  SCADA 시스템이나 Edge Controller에서 이를 실시간 수신·시각화할 수 있는 구조를 설계해야 합니다.
</p>

<h4 style={{ fontWeight: 'bold', marginTop: '24px' }}>✅ Python 코드 예시 (MQTT 전송을 통한 SRE 지표 연동)</h4>

<pre style={{ background: '#f5f5f5', padding: '16px', borderRadius: '8px', overflowX: 'auto' }}>
<code>
{`
import paho.mqtt.client as mqtt
import time
import numpy as np

# MQTT 설정
broker_address = "localhost"
topic = "sre/monitor"

client = mqtt.Client("SREPublisher")
client.connect(broker_address)

# 시뮬레이션용 SRE 값 전송
for i in range(60):
    sre_value = 0.5 + 0.3 * np.sin(i / 5.0) + np.random.normal(0, 0.05)
    payload = f"{{\\"timestamp\\": {int(time.time())}, \\"sre\\": {sre_value:.3f}}}"
    client.publish(topic, payload)
    print("📤 전송된 SRE 값:", payload)
    time.sleep(1)

client.disconnect()
`}
</code>
</pre>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  위 코드는 실시간으로 계산된 SRE 값을 <strong>MQTT를 통해 전송</strong>하여 다른 시스템이 이를 구독(Subscribe)할 수 있도록 합니다. 
  Edge 시스템에서는 이 값을 기준으로 경고 판단, 시각화, 제어 명령을 자동화할 수 있습니다. 
  SCADA 환경에서는 OPC UA 노드에 SRE 값을 실시간 등록하여 기존 RMS/FFT 값과 함께 관리할 수 있습니다.
</p>

<h3 id="labeling" style={{ marginTop: '32px', fontWeight: 'bold' }}>🏷️ 라벨 부족 문제 – 약지도 기반 평가 + 고장 시뮬레이션</h3>

<p style={{ lineHeight: '1.8' }}>
  대부분의 산업 데이터는 고장이 발생한 정확한 시점이 수동 보고서에 기반하거나 추정값에 의존하기 때문에,
  <strong>SRE 성능 평가에 필요한 정밀 라벨이 부족</strong>합니다.
  이를 해결하기 위해 <strong>약지도(Semi-supervised labeling)</strong> 구조를 도입하거나,
  실제 고장 양상을 모사하는 <strong>시뮬레이션 데이터셋</strong>을 생성하여 비교 실험을 수행할 수 있습니다.
</p>

<h4 style={{ fontWeight: 'bold', marginTop: '24px' }}>✅ Python 코드 예시 (SRE peak 기반 약지도 생성)</h4>

<pre style={{ background: '#f5f5f5', padding: '16px', borderRadius: '8px', overflowX: 'auto' }}>
<code>
{`
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# 신호 생성: 구조 리듬 붕괴 구간 포함
np.random.seed(42)
signal = np.sin(np.linspace(0, 10 * np.pi, 2000)) + 0.2 * np.random.randn(2000)
signal[1500:1600] += np.random.normal(0, 0.7, 100)  # 고장 영역 모사

# SRE 계산 (Sliding Entropy)
def compute_sre(signal, window=200, step=20, bins=32):
    sre_list, labels = [], []
    for i in range(0, len(signal) - window, step):
        seg = signal[i:i+window]
        hist, _ = np.histogram(seg, bins=bins, density=True)
        p = hist / np.sum(hist)
        p = p[p > 0]
        sre = -np.sum(p * np.log2(p))
        sre_list.append(sre)
        # 약지도 생성 기준: SRE가 이전보다 급상승할 경우 고장 사전 경고
        if len(sre_list) > 5 and (sre - sre_list[-2] > 0.8):
            labels.append(1)
        else:
            labels.append(0)
    return np.array(sre_list), np.array(labels)

sre_vals, weak_labels = compute_sre(signal)

# 시각화
plt.figure(figsize=(12, 5))
plt.plot(sre_vals, label="SRE 값")
plt.plot(weak_labels * np.max(sre_vals), 'r--', label="약지도 (경고)")
plt.title("🔍 SRE 기반 약지도 생성 예시")
plt.xlabel("슬라이딩 인덱스")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
`}
</code>
</pre>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  위 예시는 SRE 값의 변화율이 급격히 상승하는 시점을 자동으로 라벨링하여, <strong>약지도(Semi-label)</strong>로 평가 지표를 생성합니다.
  이는 정확한 고장 시점이 없는 상황에서도 일정한 기준에 따라 <strong>선행 경고 판단</strong>이 가능하도록 합니다.
  또한, SRE peak가 실제 고장보다 평균적으로 몇 초 앞서는지를 통계적으로 측정하여 <strong>선행 진단 지표로 공식화</strong>할 수 있습니다.
</p>

<h3 id="interpretability" style={{ marginTop: '32px', fontWeight: 'bold' }}>🧠 해석 가능성 부족 – 물리 기반 설명 + Rule 기반 경고 시스템</h3>

<p style={{ lineHeight: '1.8' }}>
  SRE, GPI와 같은 지표는 곡률, 엔트로피, KDE 등 수학적 개념에 기반하여, <strong>산업 실무자들에게 직관적으로 이해되기 어렵습니다.</strong>  
  이를 해결하기 위해선 <strong>M-C-K 진동 모델과 연계하여 SRE의 물리적 의미를 명확히 해석</strong>하고, 
  실제 제어에 활용 가능한 <strong>Rule 기반 의사결정 구조</strong>로 변환해야 합니다.
</p>

<h4 style={{ fontWeight: 'bold', marginTop: '24px' }}>✅ Python 코드 예시 (SRE 기준값 기반 Rule Alert 시스템)</h4>

<pre style={{ background: '#f5f5f5', padding: '16px', borderRadius: '8px', overflowX: 'auto' }}>
<code>
{`
import numpy as np
import matplotlib.pyplot as plt

# SRE 시계열 (예시 데이터)
np.random.seed(0)
sre = 0.8 + 0.2*np.sin(np.linspace(0, 10*np.pi, 200)) + 0.1*np.random.randn(200)
sre[150:170] += 0.6  # 구조 리듬 붕괴 영역

# 기준값 설정 (경고 기준)
threshold = 1.2
pre_warning_threshold = 1.0

# Rule 기반 경고 라벨
alerts = []
for val in sre:
    if val > threshold:
        alerts.append("🔴 긴급경고")
    elif val > pre_warning_threshold:
        alerts.append("🟠 주의")
    else:
        alerts.append("🟢 정상")

# 시각화
plt.figure(figsize=(12, 4))
plt.plot(sre, label="SRE 시계열", linewidth=2)
plt.axhline(y=threshold, color='r', linestyle='--', label="긴급 경고 기준")
plt.axhline(y=pre_warning_threshold, color='orange', linestyle='--', label="주의 기준")
plt.title("📋 Rule 기반 SRE 경고 구조")
plt.xlabel("시간 인덱스")
plt.ylabel("SRE 값")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
`}
</code>
</pre>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  위 구조는 실제 산업 현장에서 <strong>“SRE가 기준보다 X 이상이면 72시간 이내 베어링 교체 권고”</strong>와 같은 <strong>Rule-based 제어 전략</strong>에 바로 활용될 수 있습니다.
  또한 시각화와 함께 설명 카드(예: 감쇠 계수 증가 ➡ 진동 누적 ➡ 리듬 붕괴)를 제공하면 <strong>산업계의 신뢰와 수용성을 획기적으로 향상</strong>시킬 수 있습니다.
</p>


<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  정리하자면, SRE(Spectral Rhythm Entropy) 기반 구조 리듬 분석은 신호의 곡률과 리듬성에 기반하여 고장을 조기에 감지할 수 있는 강력한 기법이지만, 
  실제 산업 현장에 상용화되기 위해서는 다섯 가지 주요 기술적 장애물을 해결해야 합니다. 
  고차 미분 기반 구조 특성상 노이즈에 민감하며, 실시간 처리를 위한 연산 최적화가 필요합니다. 
  또한 기존 SCADA/PLC 시스템과의 연동성이 떨어지고, 정확한 고장 시점을 나타내는 라벨 부족 문제, 
  그리고 곡률, 엔트로피 등의 수학적 개념에 대한 현장 이해 부족 또한 장벽으로 작용합니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  이러한 문제들을 극복하기 위해서는 SRE-lite 모델, NSI 보완 지표, Savitzky-Golay 필터와 같은 민감도 보정 기술, 
  KDE 근사화 및 LUT 기반 경량화 구조, MQTT·OPC UA 통신 모듈 연동, 약지도 기반 진단 시스템, 
  그리고 M-C-K 물리 모델 기반 해석과 Rule 기반 경고 시스템을 도입해야 합니다. 
  궁극적으로 SRE는 물리 기반 해석력과 조기 진단 가능성을 모두 갖춘 차세대 고장 진단 지표로, 
  이론-실무 간 간극을 줄이는 핵심 기술로 자리잡을 수 있습니다.
</p>


    {/* 반응형 이미지 삽입 */}
<img
  src="/images/c4.png"
  alt="시뮬레이션 결과 이미지"
  style={{
    width: '30%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>



<h3 id="amat-endura-report" style={{ marginTop: '32px', fontWeight: 'bold' }}>
  🛠️ AMAT Endura PVD 장비 : 구조 리듬 기반 분석 적용 보고서
</h3>


<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>Applied Materials(AMAT)</b>는 미국 캘리포니아에 본사를 둔 세계 최대 반도체 장비 기업 중 하나로, 증착, 식각, 계측, 패터닝 등 다양한 공정 장비를 공급하고 있습니다.
  특히 메모리와 로직을 모두 커버할 수 있는 장비 라인업을 갖추고 있으며, 삼성전자, SK하이닉스, TSMC 등 글로벌 칩메이커들과의 긴밀한 파트너십을 유지하고 있습니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  AMAT의 대표적인 증착 장비 중 하나인 <b>Endura PVD 시스템</b>은 메탈 배선, barrier/capping layer 증착 등에서 폭넓게 활용됩니다.
  이 장비는 <b>모듈형 다중 챔버 구조</b>로 구성되어 있어 각 챔버가 독립적으로 진공을 형성하고, 증착 작업을 수행할 수 있습니다.
  또한 <b>웨이퍼 이송 로봇 암</b>, <b>증착 타겟 회전 모터</b>, <b>진공 펌프 시스템</b> 등 여러 개의 회전 구동 장치가 포함되어 있어,
  반복성과 정밀도가 높은 작업 환경을 형성하지만, 동시에 구조 리듬 분석을 적용할 수 있는 이상적인 조건을 갖추고 있습니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  Endura의 주요 동작 메커니즘은 다음과 같은 절차로 이루어집니다.
  먼저 로봇 암이 웨이퍼를 로딩하여 증착 챔버로 이송한 뒤, 진공 펌프가 수 mTorr 수준의 저압 환경을 유지합니다.
  이후 플라즈마 상태에서 금속 타겟이 원자화되어 웨이퍼 표면에 증착되고, 필요에 따라 다른 챔버로 이송되어 추가 공정을 반복합니다.
  이 일련의 과정은 높은 정밀도와 반복성을 요구하며, 각 동작에서 발생하는 진동, 전류, 압력 등의 센서 신호에는 일정한 <b>리듬 구조</b>가 내포되어 있습니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  이러한 리듬 구조는 장비가 정상적으로 작동하고 있을 때 일정한 패턴을 가지며,
  만약 베어링 마모, 중심축 편심, 회전체의 위치 이탈과 같은 미세 이상이 발생할 경우,
  해당 리듬에 변화가 생기게 됩니다. 이때 <b>SRE(Spectral Rhythm Entropy)</b>는 리듬의 복잡도 증가를, <b>GPI(Geometric Pattern Instability)</b>는 곡률 패턴의 불안정성을 통해 고장 전 조짐을 포착할 수 있습니다.
  특히 Endura 장비는 이러한 리듬 이상을 수십~수백 ms 단위로 탐지할 수 있는 대표적 사례로 볼 수 있습니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  이처럼 Endura 장비는 구조 리듬 기반 분석 기법이 효과적으로 적용될 수 있는 조건을 갖추고 있으며,
  복잡한 회전체 구조와 반복 동작, 그리고 고정밀 진공 환경이라는 특수성이 리듬 분석의 민감도를 더욱 높여줍니다.
  본 분석은 전통적인 RMS 기반 에너지 분석이나 AI 이상 탐지 기법과 달리, <b>“고장 그 자체”가 아닌 “고장이 일어나기 직전의 리듬 변화”</b>에 집중함으로써,
  조기 경고 시스템을 보다 정밀하게 구현할 수 있는 가능성을 보여줍니다.
</p>


    {/* 반응형 이미지 삽입 */}
<img
  src="/images/c5.webp"
  alt="시뮬레이션 결과 이미지"
  style={{
    width: '30%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>



<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>⚠️ 주요 고장 메커니즘:</b> Endura 장비는 모듈형 구조로 설계되어 있지만, 회전체 기반 부품에서 물리적 열화와 진동 문제가 축적됩니다. 
  Transfer Arm의 베어링은 수천 번의 왕복 동작으로 인해 금속 간 접촉 마모가 누적되며, 중심선 기준 진동 편차가 점차 커집니다. 
  증착 타겟 회전 모터는 장시간 사용 시 편심 현상이나 챔버 내 열 팽창의 영향으로 균형이 무너질 수 있으며, 이는 리듬의 정합성에 영향을 미칩니다. 
  또한 진공 펌프의 감쇠 계수 변화는 장시간 구동 시 감쇠력이 감소하여 이상 진동을 제대로 흡수하지 못하게 됩니다. 
  이러한 고장은 초기에는 수율에 영향을 주지 않지만, 누적되면 웨이퍼 스크래치, 증착 두께 불균일, 입자 오염 등의 문제로 이어집니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>📈 구조 리듬 분석의 역할:</b> SRE(Spectral Rhythm Entropy)는 시간 변화에 따라 전류나 진동 신호의 주파수 리듬이 얼마나 복잡하게 붕괴되는지를 나타내며, 
  정상 회전 상태에서는 주파수 리듬이 정돈되어 있지만 마모가 발생하면 고주파 성분이 증가하고, SRE 값이 급격히 상승합니다. 
  GPI(Geometric Pattern Instability)는 신호의 곡률 변화를 기반으로 리듬의 파편화 정도를 정량화하며, 
  편심이나 불균형 상태가 발생하면 곡률 변화가 불규칙하게 나타나며 GPI가 상승합니다. 
  SRE는 리듬의 복잡도, GPI는 균형 붕괴를 측정하는 역할을 수행합니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🧠 기존 AI 분석과의 차이:</b> 기존의 AI 기반 분석은 평균값 변화, RMS, Z-score 등을 기반으로 결과값에 대한 이상 탐지에 초점이 맞추어져 있어 고장 이후의 반응에 민감합니다. 
  하지만 GPI와 SRE는 고장 전 단계에서 구조 리듬의 붕괴를 감지할 수 있으며, 
  시계열 상에서 연속적인 위험 신호를 통해 누적된 리스크를 판단하고, 기계적 원인을 해석할 수 있는 인과 기반 시각화가 가능합니다. 
  관리자는 어느 시점부터 리듬이 무너지기 시작했는지를 직관적으로 파악할 수 있습니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🧩 실무 적용 예시:</b> Transfer Arm의 전류 신호를 2초 단위 슬라이딩 윈도우로 분석하고, 
  GPI &gt; 0.78, SRE &gt; 2.1 이상이 반복적으로 나타나는 챔버를 식별합니다. 
  해당 챔버에서 과거에 증착 두께 불균일 문제가 발생했던 기록이 수율 데이터베이스에서 확인됩니다. 
  이후 해당 베어링을 교체하자 GPI와 SRE 지표가 급감하고 수율이 회복됩니다. 
  이러한 분석은 단순 경고를 넘어 고장 원인을 직접 제거하는 인과 기반 조치로 이어질 수 있습니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🔚 결론:</b> Endura 장비는 회전 및 이송 기반 구동 시스템을 다수 포함하고 있어 구조 리듬 분석이 적용되기에 매우 적합합니다. 
  모듈별 진공 격리가 가능하다는 점에서, GPI/SRE 지표를 챔버 단위로 모니터링하여 고장 징후를 사전에 탐지할 수 있습니다. 
  이러한 구조 리듬 기반 분석은 기존의 사후 대응 위주 진단 체계를 벗어나 고장 이전 리듬 붕괴의 조기 탐지로 진화하며, 
  반도체 설비의 새로운 진단 패러다임을 실현할 수 있는 강력한 도구로 자리매김할 것입니다.
</p>


    {/* 반응형 이미지 삽입 */}
<img
  src="/images/c6.png"
  alt="시뮬레이션 결과 이미지"
  style={{
    width: '30%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>



<h3 id="asml-euv-scanner-report" style={{ marginTop: '32px', fontWeight: 'bold' }}>
  🔬 ASML EUV Scanner : 기업 및 장비 개요와 동작 메커니즘
</h3>


<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🏢 ASML: 리소그래피 기술의 절대 강자</b><br />
  네덜란드에 본사를 둔 ASML은 전 세계에서 극자외선(EUV) 리소그래피 장비를 독점 생산하는 기업입니다.
  ASML은 반도체 미세화 공정의 핵심인 나노미터급 패터닝을 구현하기 위한 광원, 광학계, 스캐닝 시스템, 진공 챔버 통합 솔루션을 제공합니다.
  특히 EUV 장비는 13.5nm 파장의 극자외선을 활용해 이전 세대 DUV(Deep UV) 장비보다 훨씬 높은 해상도의 패터닝을 실현합니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🔬 EUV Scanner 장비 구조 개요</b><br />
  ASML의 대표 장비인 NXE3400C 또는 NXE3600D와 같은 EUV Scanner는 다음의 핵심 모듈들로 구성되어 있습니다:
<ul>
  <li>1️⃣ EUV 광원 시스템 (LPP: Laser Produced Plasma)</li>
  <li>2️⃣ 진공 환경에서의 반사식 집광 미러 체계 (Bragg Reflector)</li>
  <li>3️⃣ Wafer Stage 및 Reticle Stage – 고정밀 회전·이송 구조</li>
  <li>4️⃣ Exposure Optics 및 Alignment 시스템</li>
  <li>5️⃣ Lithography Scanner Body (진공 챔버 통합)</li>
</ul>

  EUV Scanner는 단순 노광 장비가 아니라, 진공, 광학, 고속 제어, 정밀 이송, 열 안정화 등 수많은 하위 기술이 융합된 복합 시스템입니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>⚙️ 주요 동작 메커니즘</b><br />
  EUV 장비는 다음과 같은 정밀 프로세스를 수행합니다:
  <ul>
    <li><b>LPP 광원 생성:</b> 고출력 CO₂ 레이저가 틴(Tin) 드롭에 조사되며 플라즈마를 생성. 이 플라즈마에서 13.5nm 파장의 EUV가 방출됩니다.</li>
    <li><b>광 반사 및 정렬:</b> EUV는 공기 중에서 감쇠되므로 반사 거울(Bragg Mirror) 체계를 거쳐 투사됨. 굴절이 아닌 전반사로 경로를 조절합니다.</li>
    <li><b>웨이퍼 스테이지 동기화 이송:</b> 고속 진공 환경 하에서 리타이클(Reticle)과 웨이퍼를 서로 정밀하게 스캔하여 노광합니다. 이때 리니어 모터 기반의 서보 시스템, 에어베어링 또는 진공 플로팅 시스템이 사용되며, 0.1nm 수준의 위치 정밀도를 요구합니다.</li>
  </ul>
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🧠 구조 리듬 분석 가능성의 기초</b><br />
  EUV Scanner는 ‘정적인 광학 시스템’으로 보일 수 있으나, 실제로는 다음과 같은 회전·이송 구조물이 존재합니다:
<ul>
  <li>1️⃣ Wafer Stage: 수십 m/s로 가속과 정지를 반복하는 고속 슬라이드 시스템</li>
  <li>2️⃣ Reticle Stage: 회전 및 슬라이딩이 동시에 일어나는 병렬구동 이송 장치</li>
  <li>3️⃣ EUV 광원 모듈 내부의 펌프와 팬 구동계</li>
  <li>4️⃣ Alignment용 미세조정 모터 및 피에조 엑추에이터</li>
</ul>

  이러한 구성요소는 모두 정밀 반복 리듬성을 가진 구동 구조이며, 구조 리듬 기반 분석의 적용 가능성을 갖고 있습니다.
  특히 진공 내에서의 반복적 회전기기 이상은 진동과 전류 변화로 즉각 나타나므로, GPI 및 SRE의 활용 가능성이 존재합니다.
</p>


    {/* 반응형 이미지 삽입 */}
<img
  src="/images/c7.webp"
  alt="시뮬레이션 결과 이미지"
  style={{
    width: '30%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>



<h3 id="asml-euv-diagnostics-report" style={{ marginTop: '32px', fontWeight: 'bold' }}>
  🔬 ASML EUV Scanner : 고장 메커니즘 분석 및 SRE/GPI 기반 실무 적용 시나리오
</h3>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🔧 EUV 장비의 주요 고장 메커니즘</b><br />
  ASML EUV Scanner의 핵심 구성요소는 광학계 외에도 고속 이송계, 서보모터, 진공 기반 회전체 등이 있으며, 이들이 다음과 같은 고장 유형을 유발합니다:
<ul>
  <li>1️⃣ Wafer Stage 이송 시스템의 미세 정렬 오차</li>
  <li>2️⃣ 리니어 모터의 정렬 오차 또는 마찰 증가 ➡ 위치 정밀도 저하</li>
  <li>3️⃣ 고정밀 반복 구간에서 미세한 동기화 실패 ➡ 수율 저하</li>
  <li>4️⃣ Reticle Stage의 회전 불균형 또는 피로 누적</li>
  <li>5️⃣ 회전부 베어링 열화 및 감쇠 계수 증가</li>
  <li>6️⃣ 반복 주행 시 발생하는 미세 진동의 주파수 이동</li>
  <li>7️⃣ 진공 펌프 구동계 고장</li>
  <li>8️⃣ 반복적 진동/부하 증가 ➡ 진공도 불안정, 노광 품질 저하</li>
</ul>

  이러한 고장은 단순한 센서 임계치 초과로 탐지하기 어렵고, 고장의 ‘결과’가 나타나기 전의 사전 리듬 붕괴 징후가 발생합니다. 이 지점에서 SRE와 GPI의 효과가 뚜렷하게 드러납니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🧮 SRE 및 GPI 변화 흐름</b><br />
  <b>GPI (Geometric Pattern Instability) 흐름:</b><br />
  Stage 이송 중 수집된 전류/가속도 신호에서, 정상 상태일 경우 리듬 구조는 매우 정형적입니다.
  그러나 감쇠 계수 증가, 베어링 마모, 비대칭 회전 발생 시, 시계열의 구조가 급격히 불안정화되어 GPI가 급상승합니다.<br />
  예시: 정상 상태 GPI 
  <span style={{ fontFamily: 'Arial, sans-serif' }}> ≈ </span>
  0.18 
  <span style={{ fontFamily: 'Arial, sans-serif' }}> ± </span>
  0.05 / 비정상 상태 GPI 
  <span style={{ fontFamily: 'Arial, sans-serif' }}> &gt; </span>
  0.85 (비선형 곡률 구조 파편화 증가)<br /><br />

  <b>SRE (Spectral Rhythm Entropy) 흐름:</b><br />
  정상 작동 시 회전·이송 신호의 주파수 성분은 고정된 구조 리듬을 따릅니다. Stage의 정렬 불안, 진공 압력 요동 등으로 리듬이 무너짐 
  <span style={{ fontFamily: 'Arial, sans-serif' }}> ➡ </span>
  스펙트럼 분포 복잡도 상승 
  <span style={{ fontFamily: 'Arial, sans-serif' }}> ➡ </span>
  SRE 증가<br />
  예시: SRE 정상 
  <span style={{ fontFamily: 'Arial, sans-serif' }}> ≈ </span>
  1.2~1.4 / SRE 이상 
  <span style={{ fontFamily: 'Arial, sans-serif' }}> ≈ </span>
  2.0 이상 (리듬 혼돈화)
</p>



<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🛠️ 실무 적용 시나리오</b><br />
  <b>✅ 적용 위치:</b><br />
  전류 센서, 가속도 센서, 압력 센서를 통해 수집된 신호. 특히 Wafer Stage의 전류 신호와 Reticle Stage의 진동 신호는 리듬성이 뛰어나 GPI/SRE 해석에 적합합니다.<br /><br />
  <b>✅ 분석 방법:</b><br />
  5초 슬라이딩 윈도우 + 50% 오버랩 방식의 실시간 구조 분석. FFT 기반 PSD를 보조지표로 사용하여 SRE 변화와 동기 해석. GPI 상승 패턴에 대해 Bayesian Time Series Causality로 wafer 수율 저하와 인과 관계 추정.<br /><br />
  <b>✅ 실무 적용 방안:</b><br />
  기존 APM/Brightics AI와의 병렬 대시보드 구성. “Stage GPI 0.82 이상 ➡ Alignment Drift 위험 증가” 경고 메시지. 누적 GPI 증가 구간에서 수율 저하 데이터 정합 시 시각 피드백 강화.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🧠 기술적 이점</b><br />
  ⚠️ 기존 RMS/Z-score 기반 이상 감지: 특정 이상 신호만 포착<br />
  ✅ GPI/SRE 기반 분석: 고장 전조 징후를 구조적으로 감지, 특히 비주기적이고 사소한 리듬 붕괴도 정량화 가능<br />
  📉 신호 크기가 아닌 구조 복잡도 자체를 분석하므로, 미세 이상 감지에 탁월<br />
  📈 Stage 모터 열화, 베어링 마모 등 물리 기반 고장과의 직접적 연결 가능
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🎯 결론</b><br />
  ASML의 EUV 노광 장비는 고도로 정밀하지만, 그만큼 미세한 리듬 불안정성에 취약합니다.
  GPI와 SRE는 단순히 “비정상”을 넘어서, 구조의 붕괴 자체를 감지함으로써 새로운 진단 패러다임을 제시합니다.
  이는 미세공정 시대의 수율 유지와 직결되며, 현장의 실질적인 수리비용 감소와 신뢰도 향상에 직접 기여할 수 있습니다.
</p>


    {/* 반응형 이미지 삽입 */}
<img
  src="/images/c8.png"
  alt="시뮬레이션 결과 이미지"
  style={{
    width: '30%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>



<h3 id="tel-trias-overview" style={{ marginTop: '32px', fontWeight: 'bold' }}>
  🏢 Tokyo Electron (TEL) : Trias SPA<span style={{ fontFamily: 'Arial, sans-serif' }}>&#8482;</span> 플라즈마 식각기 분석 보고서
</h3>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🏭 기업 개요</b><br />
  Tokyo Electron(TEL)은 일본을 대표하는 반도체 및 디스플레이 제조 장비 기업으로, 식각, 증착, 포토레지스트 현상 등 다양한 공정 장비를 보유하고 있습니다.
  TEL은 특히 식각 및 증착 분야에서 글로벌 리더인 ASML, AMAT에 이어 세계 시장에서 높은 점유율을 확보하고 있으며,
  EUV 시대 이후 플라즈마 기반 식각 기술이 요구하는 초고정밀 제어 및 안정성 분야에서 강한 대응력을 갖추고 있습니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>⚙️ 대표 장비: Trias SPA™ 개요</b><br />
  Trias SPA는 TEL이 제공하는 고급 플라즈마 반응성 이온 식각(Plasma RIE) 장비로,
  FinFET, GAA(Gate All Around) 구조와 같이 형상비(Aspect Ratio)가 매우 높은 공정에 사용됩니다.
  미세 소자의 제작에서는 수직 식각의 정밀도와 식각 프로파일의 균일성이 매우 중요하며,
  Trias SPA는 이 분야에서 업계 최고 수준의 정밀도를 실현하는 장비로 평가받고 있습니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>⚡ 동작 원리</b><br />
  Trias SPA는 RF 전원을 통해 챔버 내 플라즈마를 생성하고, 여기에 식각용 가스를 주입하여 이온화된 플라즈마를 유도합니다.
  이 이온들은 웨이퍼 표면에 수직으로 충돌하며 식각을 수행합니다. 이때 식각 속도, 선택도, 프로파일 제어는 전류 밀도, 자기장 세기, 압력 조절, 온도 안정성에 크게 의존합니다.
  <br /><br />
  특히 식각 중 전극 및 챔버 벽에서의 전류 흐름, 플라즈마 균일도, 웨이퍼 냉각 시스템의 안정성 등은 모두 모터 및 펌프 기반 구동 장치의 반복성과 일관성에 의해 유지됩니다.
  이 과정에는 다음과 같은 주요 반복 기구가 포함됩니다:
<ul>
  <li>1️⃣ 식각 챔버 내 플라즈마 형성 및 안정화 구간</li>
  <li>2️⃣ 웨이퍼 로딩 및 언로딩 시스템 – 반복적인 이송 구조</li>
  <li>3️⃣ 하부 냉각 시스템 – 진공 펌프 및 냉각제 순환 모터</li>
  <li>4️⃣ 전극 전류 및 온도 제어를 위한 실시간 조절기</li>
</ul>

</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>📍 구조 리듬 기반 분석 가능성</b><br />
  Trias SPA 장비의 구성 요소들은 단순한 정적 제어가 아닌, 시간에 따라 주기적으로 동작하며 열, 진공, 전류, 이송을 동시에 조절하는 복합적인 구조를 가집니다.
  반복되는 웨이퍼 로딩, 냉각, 플라즈마 제어 등의 구간에서 이송 리듬성과 냉각 안정성이 공정 품질에 직접적인 영향을 미치기 때문에,
  구조 리듬 기반 분석 기법인 GPI(Geometric Pattern Instability) 및 SRE(Spectral Rhythm Entropy)의 적용 가능성이 높습니다.
  특히 전류 또는 압력 신호의 시계열을 통해 리듬 붕괴 전조를 조기 감지하면, 플라즈마 밀도 불균일, 식각 불균형, 챔버 이상 발생 등을 예방할 수 있습니다.
</p>


    {/* 반응형 이미지 삽입 */}
<img
  src="/images/c9.webp"
  alt="시뮬레이션 결과 이미지"
  style={{
    width: '30%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>



<h3 id="tel-trias-analysis" style={{ marginTop: '32px', fontWeight: 'bold' }}>
  🧪 Trias SPA<span style={{ fontFamily: 'Arial, sans-serif' }}>&#8482;</span> 식각기 : 고장 메커니즘 및 GPI/SRE 적용 분석
</h3>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🔩 고장 발생 메커니즘</b><br />
  Trias SPA는 전극, 플라즈마 발생기, 진공 펌프, 냉각 시스템 등 고정밀 부품들의 반복적 구동을 통해 정밀 식각을 수행합니다.
  이 과정에서 발생할 수 있는 대표적인 고장 유형은 다음과 같습니다:
    <br /><br />
  <ul>
    <li>🌀 진공 펌프 및 챔버 내 베어링 마모 ➡ 회전체의 미세 진동 증가 ➡ 진공 유지 불안정 ➡ 플라즈마 균질도 저하</li>
    <li>🔁 이송 장치의 정렬 오차 및 반복성 저하 ➡ 웨이퍼 로딩 시 로봇 암이 반복 패턴을 벗어남 ➡ 언로딩 실패 및 웨이퍼 손상</li>
    <li>🌡️ 온도 냉각 기구의 열교환 비효율 ➡ 동작 초반엔 정상이나 일정 시간 경과 후 온도 유지 실패 ➡ 식각 균일도 급락</li>
    <li>⚡ RF 전원 불안정 및 플라즈마 형성 편차 ➡ 플라즈마의 입자 밀도/전류값 패턴이 흐트러짐 ➡ 식각 속도 오차 발생</li>
  </ul>

    <br /><br />
  이러한 고장은 단순한 센서 임계치 초과로는 조기 감지가 어려우며,
  고장 직전의 리듬 붕괴 또는 구조의 왜곡이라는 특성으로 나타나기 때문에, GPI와 SRE와 같은 구조 기반 지표의 적용이 필수적입니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>📈 SRE 및 GPI 지표의 적용 흐름</b><br />
  <b>1️⃣ 신호 수집</b>: 기존 장비에 탑재된 전류, 압력, 진동, 온도 센서로부터 Raw 시계열 데이터를 수집하고, 반복 동작 구간(예: 로딩, 펌프 회전, 냉각 주기)으로 분할합니다.
  <br /><br />
  <b>2️⃣ Sliding Window 분석</b>: 3초 슬라이딩 윈도우 + 50% 오버랩으로 분석하며, 각 구간의 리듬을 곡률 함수 κ(t)로 변환하여 GPI를 계산하고,
  동시적으로 스펙트럼 분포의 복잡도를 기반으로 SRE를 산출합니다.
  <br /><br />
  <b>3️⃣ 조기 이상 탐지</b>:
  <ul>
    <li>GPI가 지속적으로 상승하며 곡률 파편화가 증가하는 경우 ➡ 회전기구의 피로 누적 신호</li>
    <li>SRE가 순간적으로 급상승할 경우 ➡ 리듬 붕괴 또는 반복 실패 (예: 로딩 오류, 온도 흔들림)</li>
    <li>Threshold 예: GPI &gt; 0.85, SRE &gt; 0.65에서 경고 발생</li>
  </ul>
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🛠️ 실무 적용 시나리오</b><br />
  Trias SPA는 CMP 또는 PVD 공정 직전의 전처리 장비로 사용되며, 고장 시 전체 수율에 직접 영향을 미칩니다.
  <br /><br />
  실무 적용 방안으로는 장비에서 출력되는 전류/진동 로그를 수집하고, 이를 APM(Advanced Process Management) 시스템과 연계하여 GPI/SRE를 병렬 분석합니다.
  이상 징후 발생 시 실제 수율 변화, 교체 이력, 오퍼레이터 대응 로그와의 전이 상관 분석(Granger Causality)으로 리스크를 정량화합니다.
  <br /><br />
  야간 무인지 근무 시 GPI 기반 경고 시스템을 구성하여, “GPI 상승 추세 감지 ➡ 대응 지연 위험 ➡ 경고 알람” 구조로 확장할 수 있습니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>✅ GPI/SRE 기반 진단의 장점</b><br />
<ul>
  <li>1️⃣ 기존 RMS 또는 AI 모델 대비, 고장 메커니즘에 대한 물리 기반 해석 가능</li>
  <li>2️⃣ 단순 이상 탐지를 넘어서, 반복 구간 내 구조적 리듬 붕괴의 조기 예측 가능</li>
  <li>3️⃣ Threshold 기반 정량 판단 + 리듬 시각화를 통한 정성적 진단 병행 가능</li>
</ul>

</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>⚠️ 한계 및 도전 과제</b><br />
<ul>
  <li>1️⃣ GPI는 미분 기반이므로 노이즈에 민감 ➡ 정교한 전처리 필요</li>
  <li>2️⃣ SRE는 고주파 성분에 의한 왜곡 가능성 존재 ➡ 정규화, 필터링 중요</li>
  <li>3️⃣ 신호 기반 리듬 해석에 익숙하지 않은 오퍼레이터에 대한 교육 필요</li>
</ul>

</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🔮 결론</b><br />
  Trias SPA와 같은 고정밀 플라즈마 식각 장비는 구조적으로 회전, 반복, 진동 메커니즘이 내재되어 있으며,
  GPI 및 SRE 지표는 이 장비들이 고장을 향해 이동하는 초기 징후를 조기 포착할 수 있는 고급 진단 도구입니다.
  단순한 이상 여부를 판단하는 것이 아니라, 리듬 붕괴의 동역학을 파악함으로써 고장-수율저하-장비교체로 이어지는 인과 체인을 이해할 수 있는 기반을 마련해줍니다.
</p>


    {/* 반응형 이미지 삽입 */}
<img
  src="/images/c10.webp"
  alt="시뮬레이션 결과 이미지"
  style={{
    width: '30%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>



<h3 id="lam-altus-analysis" style={{ marginTop: '32px', fontWeight: 'bold' }}>
  🔧 Lam Research : ALTUS® CVD 시스템 분석
</h3>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🏭 기업 소개: Lam Research</b><br />
  Lam Research는 미국 캘리포니아 프리몬트에 본사를 둔 반도체 제조 장비 기업으로, 특히 식각(Etch) 및 박막 증착(Deposition) 기술 분야에서 세계 최고 수준의 기술력을 보유하고 있습니다.
  아시아권 기업들이 수직적 통합을 통해 생산 효율성을 추구하는 것과 달리, Lam은 독립된 기술력으로 글로벌 Fab 고객의 고난도 공정 문제를 해결하는 역할을 맡고 있습니다.
  ALTUS 시리즈는 이러한 기술력의 집약체로, 3D NAND, DRAM, Logic 공정에서 Barrier Metal 및 High-k 물질의 증착을 정밀하게 수행하는 핵심 장비로 자리 잡고 있습니다.
  특히 ALD/CVD 하이브리드 구조를 통해 박막의 두께 제어와 수율 안정성을 동시에 구현할 수 있는 점이 주요 강점으로 작용하고 있습니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>⚙️ 주요 장비 소개: ALTUS® High-k CVD 시스템</b><br />
  ALTUS는 주로 Cobalt, Tungsten, Titanium 등 Barrier 및 Contact 메탈을 증착하는 데 사용되며, High-k 유전체 층이나 Metal Gate Stack의 형성에 있어 매우 높은 균일성과 전기적 특성이 요구되는 영역에 적용됩니다.
  이 장비는 특히 미세 공정 세대(7nm 이하)에서 성능과 수율 확보의 핵심이 되는 박막 품질 확보에 필수적입니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🧬 장비 동작 원리</b><br />
  ALTUS 시스템은 CVD(Chemical Vapor Deposition) 기반으로 작동하며, 다음과 같은 단계로 구성됩니다:
<ul>
  <li>1️⃣ Precursor 가스 주입: 금속 성분을 포함한 화학 전구체를 챔버에 주입</li>
  <li>2️⃣ 열/플라즈마 반응: 고온 또는 플라즈마 환경에서 가스 분해 ➡ 고체 금속층 형성</li>
  <li>3️⃣ 웨이퍼 로딩/언로딩: EFEM 로봇 암 + 회전 스테이지로 수백 장의 웨이퍼를 순차 처리</li>
  <li>4️⃣ 배기 및 잔류 가스 정리: 부산물 및 잔류 가스를 진공 펌프 + 스크러버로 제거</li>
</ul>

  이 전체 과정은 열 균형, 가스 플로우, 회전체의 리듬 및 이송 주기의 정밀한 동기화 위에서 작동하며,
  단 한 요소라도 리듬을 이탈하면 다음과 같은 품질 저하가 발생할 수 있습니다:
<ul>
  <li>1️⃣ 증착 불균일 (Non-uniformity)</li>
  <li>2️⃣ Gate 누설 전류 증가 (Leakage Current)</li>
  <li>3️⃣ 패턴 치수 편차 (Critical Dimension Variation)</li>
</ul>

</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🔄 반복 구동 메커니즘</b><br />
  ALTUS 시스템은 고정밀 회전체 동작과 가스/열/플라즈마의 복합 리듬이 공존하는 구조를 가지고 있습니다.
  대표적인 반복 구동 요소는 다음과 같습니다:
<ul>
  <li>1️⃣ EFEM 로봇의 이송 리듬: 지연 또는 진동 발생 시 throughput 저하 발생</li>
  <li>2️⃣ 펌프 회전 리듬: 베어링 이상 ➡ 압력 리듬 왜곡 ➡ 증착 편차 발생</li>
  <li>3️⃣ 내부 Fan, Ignition Module의 회전 및 점화 주기: 고온 플라즈마 균질도 유지 필수</li>
</ul>

  이러한 반복적이고 리듬 기반의 동작 메커니즘은 SRE(Spectral Rhythm Entropy)와 GPI(Geometric Pattern Instability)와 같은 구조 리듬 기반 분석 기법의 적용을 매우 효과적으로 수행할 수 있는 이상적인 환경을 제공합니다.
</p>


    {/* 반응형 이미지 삽입 */}
<img
  src="/images/c11.jpg"
  alt="시뮬레이션 결과 이미지"
  style={{
    width: '30%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>



<h3 id="lam-altus-failure-analysis" style={{ marginTop: '32px', fontWeight: 'bold' }}>
  ⚠️ ALTUS® CVD 시스템 : 고장 메커니즘 및 구조 리듬 분석
</h3>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>⚙️ 고장 발생 위치 및 원인</b><br />
  Lam의 ALTUS 시스템은 고정밀 회전체, 가스 주입 계통, 열 반응 메커니즘이 복합적으로 얽힌 구조로 구성되어 있으며, 다음과 같은 고장 메커니즘이 자주 발생합니다.
  첫째, CVD 반응 후 부산물을 배기하는 진공 펌프에서 <b>회전 베어링의 마모</b> 또는 유체 밀봉 실패가 발생할 경우, 회전 불균형 및 진동 증가로 이어져 챔버 내 잔류 가스량이 불안정해지고 증착 품질이 저하됩니다.
  둘째, EFEM의 로딩암은 정해진 간격으로 웨이퍼를 이송하지만, <b>모터·감속기 이상</b> 또는 센서 오류로 인해 이송 리듬이 깨지면 throughput 저하와 공정 시간 편차가 발생합니다.
  셋째, <b>Precursor 가스를 주입하는 밸브의 반복 마모</b>로 인해 초기 반응 속도가 일정하지 않게 되며, 이는 웨이퍼 간 uniformity 저하로 직결됩니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>📉 SRE/GPI 변화 양상과 해석</b><br />
  SRE(Spectral Rhythm Entropy)는 고장의 전조 단계에서 리듬의 불규칙성과 주파수 혼돈도를 측정합니다. 정상 상태에서는 SRE 값이 0.15~0.25 범위에 머무르지만, 고장 전에는 0.4 이상으로 급격히 상승하며, 이는 특히 펌프 고장 시 고주파 혼입 현상으로 인한 스펙트럼 확산 때문입니다.
  고장 후에는 SRE 값이 지속적으로 높은 수준을 유지하거나 진동성 패턴을 보이는데, 이는 리듬의 완전한 붕괴를 의미합니다.
  GPI(Geometric Pattern Instability)는 신호의 곡률 변화가 얼마나 급격하게 증가하는지를 측정하며, 구조의 리듬이 공간적으로 붕괴될 때 민감하게 반응합니다.
  GPI가 0.75 이하일 때는 정상이나, 0.8~0.9 구간의 반복 상승은 고장의 전조로 해석되며, 0.95 이상이 지속되면 고장 확률이 80%를 넘습니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🛠️ 실무 적용 시나리오 및 기대 효과</b><br />
  ALTUS 장비의 실시간 로그와 병렬로 SRE/GPI를 연동하여 분석하면, <b>기존의 AI 이상 탐지 시스템과는 다른 "리듬 기반 인과 해석"</b>이 가능합니다.
  예를 들어, 배기 펌프의 진동 리듬 변화가 감지되고 SRE 값이 급등하는 경우, GPI 역시 구조 리듬의 왜곡을 포착함으로써 "2일 내 펌프 교체 필요"라는 선제 경고를 줄 수 있습니다.
  EFEM의 로딩암 이송 간격이 불규칙해지면 throughput 저하와 GPI 상승이 동시에 관찰되며, 이는 증착 실패율 증가를 조기에 예측하는 도구로 작용할 수 있습니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🧩 장비 설계 개선으로의 확장</b><br />
  리듬 붕괴 데이터를 누적해 나가면, 반복 구동 장치의 설계 파라미터에 피드백을 제공할 수 있습니다.
  예를 들어, 로딩암의 감속기 튜닝, 플라즈마 점화 타이밍의 PID 제어 최적화, 밸브 교체 주기의 동적 조정과 같은 실무 피드백 루프를 구성할 수 있습니다.
  이는 단순한 진단을 넘어서, 설비 구조 설계 자체를 개선하는 전략적 해석 기반이 됩니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🧭 결론 – Lam ALTUS CVD에 구조 리듬 분석을 적용하는 전략적 의미</b><br />
  ALTUS와 같은 하이엔드 CVD 장비는 복합적인 리듬 기반 구조로 작동하기 때문에, 단일 센서 기반의 이상 감지보다 <b>“구조 리듬의 붕괴 전이 현상”</b>을 추적할 수 있는 분석 도구가 필요합니다.
  SRE는 시간 리듬의 혼돈도를, GPI는 공간 구조의 불안정을 수치화함으로써 고장 직전 상태를 조기에 포착할 수 있으며,
  이러한 지표는 실무자의 인지 체계에 자연스럽게 통합되어야 효과를 발휘할 수 있습니다.
  수치의 ‘언어화’, 고장 패턴의 ‘모델링’, 오퍼레이터 인지 루프의 ‘디자인’까지 통합된 전략을 주도하는 사람이 바로 이 분석을 제안하는 당신이 될 수 있습니다.
</p>


    {/* 반응형 이미지 삽입 */}
<img
  src="/images/c12.jpg"
  alt="시뮬레이션 결과 이미지"
  style={{
    width: '30%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>


<h3 id="kla-surfscan-mechanism" style={{ marginTop: '32px', fontWeight: 'bold' }}>
  🔍 KLA Surfscan SP 시리즈 : 장비 개요 및 동작 메커니즘
</h3>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🏢 회사 및 장비 개요</b><br />
  KLA(KLA Corporation)는 반도체 공정에서 <b>결함 검사(Defect Inspection)</b> 및 계측(Metrology) 분야의 세계적 리더로, 광학 기반의 결함 탐지 및 치수 측정 장비를 포함한 다양한 솔루션을 제공합니다.
  Surfscan SP 시리즈는 광학 스캐닝 기반의 고정밀 웨이퍼 검사 장비로, 파티클, 스크래치, 필름 이상 등을 수 nm 단위로 검출할 수 있습니다.
  이 장비는 300mm 웨이퍼까지 대응하며, <b>Darkfield/Brightfield 모드</b> 전환, 고속 회전 기반의 나선형 스캐닝, 1nm 이하 결함 감지 능력 등을 갖추고 있습니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>⚙️ 주요 동작 메커니즘</b><br />
  검사 대상 웨이퍼는 EFEM 로봇에 의해 로딩되며, 중앙 축에 고정된 뒤 수천 RPM의 속도로 회전합니다.
  회전 중 광원에서 발사된 빛이 웨이퍼 표면에 입사되고, 반사 또는 산란된 광을 검출기가 감지하며 결함을 분석합니다.
  이때 회전과 광선 주사선은 정밀하게 동기화되어야 하며, 검사 범위는 외곽에서 중심까지 나선형으로 확장됩니다.
  광학 모듈은 표면의 미세한 높이 차이를 실시간으로 보정하기 위해 Z축 제어 알고리즘을 기반으로 빠르게 렌즈 위치를 조정합니다.
  또한 EFEM 로딩암은 반복적인 주기로 웨이퍼를 반송하며, 각 과정은 고정밀 리듬으로 구성되어 검사 정확도를 유지합니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>💥 고장이 발생할 수 있는 위치 및 원인</b><br />
  첫째, <b>웨이퍼 회전축의 진동 이상</b>이 발생할 경우 베어링 마모나 축 중심의 미세 틀어짐으로 인해 회전이 비대칭화되어, 광학 스캔 신호에 왜곡이 발생하고 노이즈가 증가합니다.
  둘째, <b>광학 모듈의 Z축 피드백 제어</b>가 불안정할 경우 렌즈의 보간 제어에 고주파 진동이 유입되어 검사 이미지에 오류가 발생할 수 있습니다.
  셋째, <b>EFEM 로딩암의 리듬 붕괴</b>는 반복 반송 리듬 내 미세한 시간 지연이 누적될 경우 공정 병목 및 throughput 저하를 유발합니다.
  Surfscan 장비는 회전, 광학, 이송 리듬이 정밀하게 맞물리는 구조이므로, 어느 하나라도 리듬이 붕괴되면 전반적인 검사 정확도에 심각한 영향을 미치게 됩니다.
</p>


    {/* 반응형 이미지 삽입 */}
<img
  src="/images/c13.jpeg"
  alt="시뮬레이션 결과 이미지"
  style={{
    width: '30%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>



<h3 id="kla-surfscan-sre-gpi" style={{ marginTop: '32px', fontWeight: 'bold' }}>
  📈 SRE/GPI 기반 분석 적용 흐름 및 실무 가능성
</h3>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🔍 구조 리듬 기반 분석의 적용 필요성</b><br />
  Surfscan SP 시리즈는 회전, 주사, 보간, 반송 등의 각 구성 요소가 정밀한 시간 및 공간 리듬으로 맞물려 작동하는 장비입니다.
  이 리듬 구조가 무너지면 결함 검출의 정밀도가 떨어지며, 기존의 이상 탐지 방식은 주로 결과 기반 진단이기 때문에 사후적인 경향이 강합니다.
  반면, <b>SRE(Spectral Rhythm Entropy)</b>와 <b>GPI(Geometric Pattern Instability)</b>는 리듬 구조 자체의 붕괴를 실시간으로 조기에 감지할 수 있습니다.
  SRE는 회전 리듬의 주파수 혼돈도를 엔트로피로 정량화하고, GPI는 곡률 변화의 급변 양상을 통해 회전축 진동이나 이송 불균형을 포착합니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>💡 적용 사례 1: 회전축 마모 조기 감지</b><br />
  베어링의 미세 마모가 시작되면 전류 진동과 광학 리턴파 패턴에서 고주파 비대칭이 발생합니다.
  이때 SRE는 고주파 대역에서 스펙트럼이 퍼지며 엔트로피 값이 증가하는 양상을 보입니다.
  GPI는 회전 속도 변화에 따라 발생하는 미세 곡률 왜곡을 실시간으로 수치화하여 포착합니다.
  두 지표 모두 기존 RMS, FFT, Z-score 기반 이상 탐지보다 더 이른 시점에 경고 신호를 제공합니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>⚙️ 적용 사례 2: Z축 렌즈 피드백 불안정</b><br />
  렌즈 위치 보정용 액추에이터가 마모되거나 제어 전류에 노이즈가 포함되면 진동 리듬이 왜곡됩니다.
  GPI는 시간축에서 곡률의 기울기 변화율이 급격히 요동치는 구간을 감지하고, 이는 제어 루프의 동적 불안정성과 직접적으로 연결됩니다.
  SRE는 전류 진동의 주파수 혼합도가 증가하는 것을 통해 신호 복잡도 상승을 정량화합니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>📊 실무 적용 시나리오</b><br />
  검사 주기 동안 센서 및 전류 데이터를 실시간 수집한 후, 슬라이딩 윈도우 방식으로 SRE(t), GPI(t)를 계산합니다.
  일정 기준 이상으로 지표가 상승하면 즉각 경고를 발생시킬 수 있으며, 예시로 GPI &gt; 0.72이면 회전축 마모가 의심되고,
  SRE &gt; 1.15이면 렌즈 제어계의 불안정 가능성이 제기됩니다.
  이를 ‘리듬 경고 대시보드’에 시각화하여 관리자에게 실시간 피드백을 제공하고, 기존 AI 이상 탐지 체계와 병렬로 동작할 수 있습니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>✅ 장점</b><br />
  조기 감지 능력이 뛰어나 고장이 발생하기 이전의 징후를 미리 포착할 수 있으며,
  리듬 구조에 기반한 설명이 가능하므로 설비 엔지니어에게 높은 신뢰도를 제공합니다.
  검사 정확도의 저하나 불량률 증가를 사전에 방지할 수 있는 실질적인 예방 효과도 기대할 수 있습니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>⚠️ 단점</b><br />
  실시간 GPI 계산은 고차 미분 연산이 요구되기 때문에 연산 부하가 크며, 고속 데이터 처리 환경이 필수적입니다.
  또한 기존 RMS, FFT 방식에 비해 개념이 복잡하여 현장 실무자 대상의 해석 교육이 병행되어야 합니다.
  Surfscan 시스템의 기존 로그 시스템과의 연동이 어렵기 때문에 MES, 고장 DB, 수율 맵 등과의 데이터 통합도 필요합니다.
</p>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>🧭 결론</b><br />
  KLA Surfscan SP는 구조적 리듬이 핵심 성능 요소로 작용하는 검사 장비입니다.
  GPI와 SRE는 이러한 리듬이 처음부터 무너지기 시작하는 시점을 정량적으로 감지할 수 있는 도구이며,
  기존 이상 탐지 체계로는 감지하기 어려운 '전조 상태'를 실시간으로 추적할 수 있습니다.
  이는 단순 고장 탐지를 넘어 수율 안정화와 정밀도 유지의 핵심 전략으로 진화할 수 있으며,
  향후 KLA 장비군에서 구조 리듬 분석 기법의 도입은 필연적인 선택이 될 가능성이 높습니다.
</p>

<h3 id="sre-gpi-economic-scenarios" style={{ marginTop: '32px', fontWeight: 'bold' }}>
  💰 SRE/GPI 기술의 경제적 가치: 긍정적 vs 부정적 시나리오
</h3>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>📊 개요: 경제적 가치 평가의 필요성</b><br />
  구조 리듬 기반 분석 기술(SRE/GPI)은 반도체 산업에서 장비의 예지 보전을 실현하는 핵심 기술로 부상할 수 있습니다.
  다운타임 감소, 수율 향상, 유지보수 효율화, 자산 수명 관리 등 다양한 측면에서 경제적 가치를 창출할 수 있으며,
  이에 따라 실제 반도체 FAB에서의 도입 시나리오를 긍정적/부정적으로 구분해 수치 기반의 영향력을 분석하는 것이 필요합니다.
</p>

<h4 style={{ fontWeight: 'bold', marginTop: '24px' }}>✅ 긍정적 시나리오: 성공적 도입 및 확산</h4>

<p style={{ lineHeight: '1.8', marginTop: '12px' }}>
  <b>🕛 다운타임 감소 효과</b><br />
  장비 고장으로 인한 비계획 정지는 대형 FAB에서 막대한 손실을 초래합니다.
  SRE/GPI 기술이 고장 수 시간~수일 전의 리듬 붕괴를 감지함으로써 예지 정비가 가능해지고,
  장비 가동 중단 시간이 30%만 줄어들어도 연간 약 100억 원 이상의 손실 방지가 가능합니다.
</p>

<p style={{ lineHeight: '1.8' }}>
  <b>📈 수율 향상 효과</b><br />
  웨이퍼 생산 수율이 0.1%만 향상되더라도, 월 10만장 생산 기준으로 연간 약 300억 원 이상의 수익 증대 효과가 발생합니다.
  이는 구조 리듬 기반 이상 탐지를 통해 불량 유발 요인을 조기 제거할 수 있기 때문입니다.
</p>

<p style={{ lineHeight: '1.8' }}>
  <b>💰 유지보수 비용 절감</b><br />
  기존 유지보수 방식은 고장 후 긴급 수리나 불필요한 정기 점검으로 인해 비효율적인 자원 낭비가 발생합니다.
  SRE/GPI 기반 예측 유지보수 체계는 유지보수 비용의 20~30% 절감을 가능하게 하며, 이는 연간 100~150억 원 절감으로 이어질 수 있습니다.
</p>

<p style={{ lineHeight: '1.8' }}>
  <b>🦾 장비 자산 수명 연장</b><br />
  회전체의 마모나 편심, 진공 펌프의 감쇠 저하 등을 조기에 감지하여 구조적 손상을 예방할 수 있습니다.
  이를 통해 고가 장비의 교체 시기를 1~2년 이상 연장할 수 있으며, 100대 규모의 장비군에서 수백억 원의 자산 보존 효과를 기대할 수 있습니다.
</p>

<p style={{ lineHeight: '1.8' }}>
  <b>🔁 장비 설계 피드백 및 신제품 개발 기여</b><br />
  SRE/GPI 기술로 축적된 리듬 데이터는 장비 설계의 약점을 식별하는 중요한 피드백으로 작용할 수 있습니다.
  이는 장비 제조사와의 공동 개발로 이어지며, 차세대 장비의 신뢰성과 경쟁력을 높이는 기반이 됩니다.
</p>

<p style={{ lineHeight: '1.8' }}>
  <b>📊 종합 효과</b><br />
  위 요소들을 종합하면, SRE/GPI 기술의 성공적 도입 시 연간 700억 원에서 1,000억 원 이상의 직접적인 경제적 가치를 창출할 수 있으며,
  삼성전자, SK하이닉스, TSMC와 같은 Tier-1 기업에서는 이 수치가 연간 수천억 원 규모로 확대될 수 있습니다.
</p>

<h4 style={{ fontWeight: 'bold', marginTop: '32px' }}>❌ 부정적 시나리오: 도입 실패 또는 제한적 확산</h4>

<p style={{ lineHeight: '1.8', marginTop: '12px' }}>
  <b>💸 초기 도입 비용 과다</b><br />
  SRE/GPI 시스템을 도입하기 위해서는 고정밀 센서 설치, 신호 수집 장치 구축, 분석 서버 운영 등 초기 인프라 투자 비용이 필요합니다.
  이는 공정별로 수십억 원 규모가 소요되며, 도입 시 ROI 확보에 대한 의문이 발생할 수 있습니다.
</p>

<p style={{ lineHeight: '1.8' }}>
  <b>⚠️ 신뢰성 확보 실패</b><br />
  경고 이벤트와 실제 고장 간의 불일치가 반복될 경우, 기술에 대한 신뢰도가 하락하고 현장 엔지니어들 사이에서 거부감이 생길 수 있습니다.
  이는 유지보수 체계의 혼선을 유발하고, 오히려 수율 저하나 불필요한 정비로 이어질 위험도 있습니다.
</p>

<p style={{ lineHeight: '1.8' }}>
  <b>📉 기술 고립 및 운영 충돌</b><br />
  기존의 RMS/FFT 기반 진단 체계와의 통합이 어렵거나, 신호 해석 방식의 표준화가 되지 않을 경우 기술은 고립될 수 있습니다.
  그 결과로 기존 시스템과 병렬 운영이 필요해지며, 중복 비용이 발생하고 운영 복잡도가 증가합니다.
</p>

<p style={{ lineHeight: '1.8' }}>
  <b>🚫 장비사와의 협력 실패</b><br />
  리듬 데이터의 정확한 해석을 위해 장비 내부 구조나 설계와의 연동이 필요한데, 장비사 측에서 이를 거부하거나 협력하지 않는다면
  기술 확산이 제한되고 장기적으로 기술이 폐기될 가능성도 존재합니다.
</p>

<p style={{ lineHeight: '1.8' }}>
  <b>📉 종합 손실 가능성</b><br />
  위와 같은 요소가 동시에 발생할 경우, 연간 100억 원에서 최대 300억 원까지의 손실 및 기회비용이 발생할 수 있으며,
  장기적으로는 기술 자체의 도입 가능성이 산업 내에서 사라질 수 있는 리스크도 존재합니다.
</p>

<h4 style={{ fontWeight: 'bold', marginTop: '32px' }}>📌 결론: 기술 도입의 전략적 판단 기준</h4>

<p style={{ lineHeight: '1.8' }}>
  SRE/GPI 기술은 다운타임 감축, 수율 향상, 설비 자산 최적화 측면에서 매우 높은 경제적 파급력을 가지고 있습니다.
  그러나 초기에 신호 해석의 정확도와 실무 적용성 확보가 실패할 경우, 기술은 쉽게 외면받을 수 있는 구조적 한계도 존재합니다.
  따라서 <b>Pilot 적용 ➡ 효과 검증 ➡ 점진 확산</b>이라는 단계적 도입 전략과, 신뢰 가능한 수치 기반의 성과 분석이 병행되어야
  산업 전반으로 확산 가능한 고부가 기술로 자리잡을 수 있습니다.
</p>

<h3 id="samsung-sk-line-application" style={{ marginTop: '32px', fontWeight: 'bold' }}>
✅ 1. 적용 대상: 삼성전자 & SK하이닉스 라인 구조
</h3>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
<b>📌 삼성전자</b>는 평택 P3 라인(5nm EUV, DRAM), 화성 라인, 온양 패키징 라인을 주력으로 운영하며,
ASML EUV, Lam ALTUS, TEL Trias, AMAT Endura, KLA Surfscan 등 고정밀 회전체 장비를 다수 보유하고 있습니다.
노광, 증착, 식각, 클린, CMP, 패키징 등의 공정에서 회전체 기반 장비가 집중되어 있어 구조 리듬 기반 진단 기술의 적용 여지가 높습니다.
전류 및 진동 센서가 부착된 장비가 다수 존재하며, Top-down 방식으로 PoC 승인을 받은 후 확산되는 방식의 문화적 특성을 가집니다.
</p>

<p style={{ lineHeight: '1.8' }}>
<b>📌 SK하이닉스</b>는 이천 M16과 청주 M15 라인에서 DRAM 제조를 집중적으로 수행하고 있으며,
이송 및 패키징, 후공정 검사, 로딩 유닛 등 자동화된 장비군에서 회전체 기반의 구조 리듬 분석이 가능할 것으로 예상됩니다.
Lam, AMAT, TEL 등의 장비와 자체 커스터마이징 장비를 혼용하고 있으며,
Bottom-up 제안보다는 실증 중심의 도입을 선호하는 실무 중심의 문화가 특징입니다.
</p>

<h3 id="realistic-application-scenario" style={{ marginTop: '32px', fontWeight: 'bold' }}>
⚙️ 2. 현실적 적용 시나리오 흐름 (PoC~확산)
</h3>

<h4 style={{ fontWeight: 'bold', marginTop: '16px' }}>1️⃣ 파일럿 단계 (6개월 이내)</h4>
<ul style={{ lineHeight: '1.8' }}>
  <li><b>대상 장비 선정:</b> Lam ALTUS CVD, AMAT Endura PVD 등 회전체 및 펌프 기반 장비</li>
  <li><b>신호 확보:</b> 설비 내 전류/진동 센서를 Logger 또는 PLC에서 미러링 수집</li>
  <li><b>SRE/GPI 도입:</b> GPI는 이송 장비의 곡률 변화, SRE는 CVD 배기계의 리듬 주파수 기반 분석</li>
  <li><b>고장 전조 탐지 여부 확인:</b> 과거 고장과의 정합성 검증 (False alarm 최소화)</li>
  <li><b>산출물:</b> 고장 사전 감지율 70% 이상, 수율 변화 correlation 분석, 운영 리포트 제출</li>
</ul>

<p style={{ lineHeight: '1.8' }}>
🟢 <b>예상 성과:</b><br />
다운타임 10~20% 감소<br />
수율 저하 연계 고장 조기 인지 성공<br />
단일 장비군 기반 운영 ROI 가시화 시작
</p>

<h4 style={{ fontWeight: 'bold', marginTop: '24px' }}>2️⃣ 부서 단위 확산 (6~12개월)</h4>
<ul style={{ lineHeight: '1.8' }}>
  <li><b>데이터 기반 확장:</b> 1차 PoC에서 수집한 SRE/GPI 데이터 확대 분석</li>
  <li><b>운영 정책 변경:</b> 예측 이벤트 발생 시 인력 자동 배치 및 점검 주기 최적화</li>
  <li><b>시각화 연동:</b> FDC 및 Fab Dashboard 시스템에 SRE/GPI 경고 통합</li>
  <li><b>이슈 대응:</b> False Positive 최소화를 위한 NSI 병렬 구조 도입</li>
  <li><b>조직 내 신뢰 확보:</b> 실장급 이상 보고, 구조적 해석 기반의 논리 공유</li>
</ul>

<p style={{ lineHeight: '1.8' }}>
🟢 <b>기대 성과:</b><br />
장비 전체 가동률 1~2% 상승<br />
수율 개선 효과 연간 수십억 원 규모 확인<br />
유지보수 비용 10% 절감 및 고부가 인력 전환
</p>

<h4 style={{ fontWeight: 'bold', marginTop: '24px' }}>3️⃣ FAB 전반/기업 차원 확산 (2~3년)</h4>
<ul style={{ lineHeight: '1.8' }}>
  <li><b>디지털 트윈 연동:</b> GPI/SRE 지표를 디지털 트윈 시뮬레이터에 연계</li>
  <li><b>장비사와 공동 개선:</b> Lam, KLA 등과 리듬 붕괴 기반의 구조 설계 개선 추진</li>
  <li><b>엔지니어링 표준화:</b> PM 매뉴얼 내 SRE/GPI 기준치 삽입</li>
  <li><b>통합 모니터링:</b> RMS/FFT 기반 체계와 병렬 연동된 시각화 완비</li>
  <li><b>수율 AI 모델 활용:</b> 고장 조기 진단 및 품질 예측용 피처로 SRE/GPI 활용</li>
</ul>

<p style={{ lineHeight: '1.8' }}>
🟢 <b>최종 기대 성과:</b><br />
다운타임 최대 30% 감소<br />
수율 개선 연간 수백억 원 규모<br />
신기술 적용 FAB 벤치마크로 글로벌 경쟁력 확보
</p>

<h3 id="economic-estimation" style={{ marginTop: '32px', fontWeight: 'bold' }}>
💰 3. 경제적 가치 추정 (삼성전자 DRAM 기준)
</h3>

<p style={{ lineHeight: '1.8' }}>
<b>보수적 추정:</b><br />
수율 개선(0.1%)으로 +150억 원/년<br />
가동률 상승(1~2%)으로 +50~100억 원/년<br />
유지보수 절감 +20~50억 원/년<br />
고장 사전 방지 +30~100억 원/년<br />
<b>총계:</b> 연간 +250~400억 원 수준
</p>

<p style={{ lineHeight: '1.8' }}>
<b>적극적 추정:</b><br />
수율 개선 +400억 원/년<br />
가동률 상승 +200억 원/년<br />
유지보수 절감 +100억 원/년<br />
고장 사전 방지 +100억 원/년<br />
<b>총계:</b> 연간 +600~800억 원 이상 (DRAM FAB 단일 기준)<br />
📌 이 수치는 삼성 평택 P3 또는 SK하이닉스 M16 FAB 기준 추정치이며, 전사 확산 시 수천억 원 규모로 확대될 수 있습니다.
</p>

<h3 id="risk-and-response" style={{ marginTop: '32px', fontWeight: 'bold' }}>
🧠 4. 도입 시 리스크 및 대응 전략
</h3>

<ul style={{ lineHeight: '1.8' }}>
  <li><b>기술 신뢰 부족:</b> 신호 해석이 추상적일 경우 거부감 유발 ➡ GPI 곡률 기반 논문화 + 설명회 병행</li>
  <li><b>실시간성 미흡:</b> 고속 설비에서는 데이터 처리 지연 ➡ ESP-Lite + NSI 병렬 구조로 민감도 제어</li>
  <li><b>시스템 통합 비용:</b> FDC, MES 연동 비용 부담 ➡ 선택적 장비군부터 점진적 확산 설계</li>
  <li><b>False Positive 이슈:</b> 불필요한 점검 유발 ➡ Threshold 튜닝 및 구조 조건 동시 적용</li>
</ul>

<h3 id="conclusion" style={{ marginTop: '32px', fontWeight: 'bold' }}>
✅ 결론
</h3>

<p style={{ lineHeight: '1.8' }}>
삼성전자와 SK하이닉스 생산 라인에 SRE/GPI 기술을 적용할 경우, 연간 수백억 원 규모의 수익 향상 및 장비 가동률 개선 효과가 기대됩니다.
기술 도입 초기에는 PoC 기반의 신뢰성 확보와 실증 성공이 핵심이며, 해석 가능성과 시각화 연동 역량이 확산의 핵심 열쇠가 됩니다.
신뢰 기반의 성과 지표를 축적한다면, 해당 기술은 반도체 제조업 전반의 디지털화 흐름 속에서 전략 기술로 자리매김할 수 있습니다.
</p>

<h3 id="sk-hynix-application" style={{ marginTop: '32px', fontWeight: 'bold' }}>
🏢 SK하이닉스 전용 SRE/GPI 기술 적용 시나리오
</h3>

<p style={{ lineHeight: '1.8', marginTop: '16px' }}>
  <b>📌 주요 적용 대상 라인 및 특성</b><br />
  SK하이닉스는 이천 M16과 청주 M15 라인에서 고성능 DRAM 생산을 주도하고 있으며, 특히 후공정(이송, 패키징, 검사) 자동화 비중이 높습니다.
  Lam, TEL, AMAT 계열의 장비 외에도 자체 커스터마이징된 이송 장비 및 진공 공정 장비가 많아, 곡률 기반 구조 리듬 분석 기술(GPI/SRE)의 적용성이 높습니다.
  실무 중심의 <b>Bottom-up 실증 문화</b>가 강하며, 단일 공정에서의 성능 검증을 기반으로 기술을 확산시키는 경향이 강합니다.
</p>

<h4 style={{ fontWeight: 'bold', marginTop: '24px' }}>⚙️ 파일럿 도입 시나리오 (6개월)</h4>
<ul style={{ lineHeight: '1.8' }}>
  <li><b>적용 장비:</b> 로딩 암, 진공 배기 펌프, 검사기 이송 유닛</li>
  <li><b>신호 수집:</b> 기존 전류/진동 센서 I/F 또는 추가 센서 부착 후 로컬 Logger 연동</li>
  <li><b>분석 방식:</b> GPI는 이송 경로의 곡률 변화, SRE는 배기 계통의 주파수 불안정성 분석</li>
  <li><b>평가 지표:</b> 고장 사전 감지율, 수율 변화와의 상관성, FDC 연동 가능성</li>
</ul>

<p style={{ lineHeight: '1.8' }}>
🟢 <b>기대 효과:</b> 고장 예지 성능 70% 이상, 비계획 정비 최소화, 라인 수율 저하 선제 대응 가능성 확인
</p>

<h4 style={{ fontWeight: 'bold', marginTop: '24px' }}>📈 부서 단위 확산 (6~12개월)</h4>
<ul style={{ lineHeight: '1.8' }}>
  <li><b>적용 영역 확대:</b> 후공정 자동화 장비군으로 확대</li>
  <li><b>정비 자동화 연동:</b> GPI 경보 발생 시 정비 일정 자동 등록</li>
  <li><b>운영 시스템 통합:</b> SRE/GPI 지표를 Fab Dashboard 또는 Yield Monitor와 연동</li>
  <li><b>조직 내부 확산:</b> 실장/파트장 주도 정기 리뷰로 기술 내재화</li>
</ul>

<p style={{ lineHeight: '1.8' }}>
🟢 <b>성과 기대:</b> DRAM 라인 전체에서 연간 수율 0.1~0.3% 개선, 다운타임 20% 이상 감소
</p>

<h4 style={{ fontWeight: 'bold', marginTop: '24px' }}>🏭 전사 확산 및 전략적 정착 (2~3년)</h4>
<ul style={{ lineHeight: '1.8' }}>
  <li><b>PM 기준 정비:</b> 구조 리듬 이상 조건을 정기 점검 항목에 반영</li>
  <li><b>설비별 알고리즘 최적화:</b> 장비 유형별 GPI/SRE 학습 기반 Threshold 자동 설정</li>
  <li><b>신뢰성 인증 확보:</b> SRE/GPI 기반 정비 이력으로 장비 성능 추적</li>
  <li><b>협력사 피드백:</b> 장비 제작사에 리듬 붕괴 패턴 공유 ➡ 차세대 장비 개선 피드백</li>
</ul>

<p style={{ lineHeight: '1.8' }}>
🟢 <b>장기적 기대:</b> 예지 정비가 표준으로 정착되며, 수백억 원 단위의 장비 교체 비용 절감 및 라인 수율 안정화 달성
</p>

<h4 style={{ fontWeight: 'bold', marginTop: '24px' }}>💡 SK하이닉스 특화 전략 요약</h4>
<ul style={{ lineHeight: '1.8' }}>
  <li><b>실증 기반 확산:</b> 하나의 장비/공정에서 성능 입증 ➡ 수직 계열 부서 전체 확산</li>
  <li><b>시각화 기반 보고:</b> Dash 패널을 통해 관리직/임원 대상 고장 위험도 실시간 공유</li>
  <li><b>MES/FDC 친화형 설계:</b> 기존 Yield/Alarm 시스템과의 연동 고려한 지표 설계</li>
</ul>

<p style={{ lineHeight: '1.8' }}>
📌 SK하이닉스는 구조 리듬 분석 기술을 통해 실무 중심의 정비 자동화, 수율 개선, 설비 수명 관리라는 세 가지 핵심 지표를 모두 강화할 수 있으며,
라인의 복잡도와 자동화 수준이 높을수록 이 기술의 도입 효과는 더욱 증대됩니다.
</p>

    </div>
  );
}
