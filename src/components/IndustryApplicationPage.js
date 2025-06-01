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
  이는 기존 AI 기반 이상 탐지와 전통 RMS 기반 방식 사이의 중간지대 기술로서, **설명 가능하면서도 실시간 경고가 가능한 고장 진단 플랫폼으로 발전할 가능성이 매우 큽니다.**
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
  또한 시각화와 함께 설명 카드(예: 감쇠 계수 증가 → 진동 누적 → 리듬 붕괴)를 제공하면 <strong>산업계의 신뢰와 수용성을 획기적으로 향상</strong>시킬 수 있습니다.
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


    </div>
  );
}
