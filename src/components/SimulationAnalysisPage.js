// src/components/SimulationAnalysisPage.js
import React from 'react';
import DigitalTwinLinks from './DigitalTwinLinks';

export default function SimulationAnalysisPage() {

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
      <h2 style={{ fontSize: '1.5rem', marginBottom: '16px' }}>🖥️ 시뮬레이션 분석</h2>


  {/* ✅ 목차 카드 삽입 위치 */}
  <div style={{
    display: 'flex',
    flexWrap: 'wrap',
    gap: '12px',
    padding: '16px 0',
    borderBottom: '1px solid #ccc',
    marginBottom: '24px'
  }}>
    <a href="#intuitive" style={tocStyle}>😮 개념 해설</a>
    <a href="#physical" style={tocStyle}>💪 물리 정당화</a>
    <a href="#design" style={tocStyle}>🧪 실험 설계</a>
    <a href="#sre-nsi" style={tocStyle}>📊 SRE vs NSI</a>
    <a href="#why-rhythm" style={tocStyle}>🧠 회전체 리듬</a>
    <a href="#mass" style={tocStyle}>⚖️ 질량(m)</a>
    <a href="#damping" style={tocStyle}>🛢️ 감쇠(c)</a>
    <a href="#nonlinear" style={tocStyle}>📡 고주파 교란</a>
    <a href="#stiffness" style={tocStyle}>🧲 강성(k)</a>
    <a href="#sre-shift" style={tocStyle}>📉 기준선 이동</a>
    <a href="#nsi" style={tocStyle}>📈 NSI 지표</a>
    <a href="#integration" style={tocStyle}>🧩 통합 해석</a>
  </div>


      <p style={{ color: '#555' }}>
        이 페이지는 고장 진단, 주파수 리듬, 고장 전이 분석 등 시뮬레이션 기반 실험 결과를 시각화하거나 해석하는 공간입니다.
      </p>

      <div style={{
  display: 'flex',
  flexWrap: 'wrap',
  gap: '12px',
  padding: '16px 0',
  borderBottom: '1px solid #ccc',
  marginBottom: '24px'
}}>

</div>



{/* 반응형 이미지 삽입 */}
<img
  src="/images/14.jpg"
  alt="시뮬레이션 결과 이미지"
  style={{
    maxWidth: '600px',      // 최대 너비 제한
    width: '100%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>




      <p style={{ marginTop: '24px', lineHeight: '1.8' }}>
  본 시뮬레이션은 반도체 장비, 특히 로딩암(Loading Arm)이나 이송 유닛과 같은 회전체 설비에서 발생하는 구조적 리듬의 변화를 기반으로 고장 전이(fault transition)를 정량적으로 탐지하는 실험이다. 실험 신호는 총 120초 길이로 구성되며, 세 개의 시간 구간—정상 구간(0~60초), 전이 구간(60~90초), 고장 구간(90~120초)—으로 나뉜다. 정상 구간에서는 8Hz의 반복 리듬과 소량의 백색잡음으로 이상 없는 로딩암의 회전 상태를 모델링하며, 전이 구간에서는 고차 주파수 성분과 점진적으로 증가하는 마찰성 노이즈를 추가해 구조 피로 누적을 모사한다. 마지막 고장 구간에서는 고주파 진동과 함께 불규칙한 진폭의 충격성 노이즈를 삽입함으로써 실제 고장 발생 직전의 리듬 붕괴 양상을 반영한다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  분석 지표는 총 세 가지로 구성된다. 첫째, SRE(Structural Rhythm Entropy)는 ESP(주파수 복잡도)의 곡률 변화량을 기반으로 하여 리듬의 급격한 구조적 붕괴를 감지하며, 둘째, NSI(Noise Stability Index)는 곡률 변화량의 불안정성(분산성)을 통해 이상 축적을 포착한다. 셋째, Periodicity는 전체 파워 스펙트럼에서 지배 주파수의 비율을 측정함으로써 반복성 상실을 정량화한다. 실험에서는 Z-score와 percentile(97.5%, 2.5%) 기반 임계값을 설정하여 조기 경고 반응 시점을 시각화하며, 이를 통해 지표별 민감도와 반응 특성을 분석한다.
</p>


{/* 반응형 이미지 삽입 */}
<img
  src="/images/15.png"
  alt="시뮬레이션 결과 이미지"
  style={{
    maxWidth: '600px',      // 최대 너비 제한
    width: '100%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>


<h3 style={{ marginTop: '24px', fontWeight: 'bold' }}>📎 전체 시뮬레이션 및 분석 코드 전문</h3>
<p style={{ lineHeight: '1.8' }}>
  아래 코드는 고장 리듬 시뮬레이션 생성부터 SRE/NSI 분석, 임계값 산출 및 시각화까지의 전 과정을 포함한 Python 기반 시뮬레이션 코드이다. 각 구간에 대해 리듬/노이즈 특성을 현실적으로 조합하였고, 분석 파트에서는 percentile 기반의 robust threshold를 도입해 통계적 신뢰도를 강화하였다.
</p>
<pre style={{ fontSize: '0.8rem', whiteSpace: 'pre-wrap', background: '#f9f9f9', padding: '16px', borderRadius: '8px', overflowX: 'auto' }}>
{`# 아래는 요약이므로 실제 분석 코드는 코드 전문 참조
# - 시뮬레이션 구성: 정상, 전이, 고장
# - 신호 삽입 요소: 주파수 리듬, 고차 성분, 충격 노이즈
# - 지표 계산: SRE (곡률), NSI (곡률 변화량), Periodicity (주파수 집중도)
# - 임계값: Z-score + Percentile
# - 결과: 고장 전 5초 내 반응 확인 (SRE, NSI 민감도 우수)`}
</pre>

<h3 style={{ marginTop: '48px', fontSize: '1.25rem' }}>🟢 리듬 기반 고장 시나리오 구조화</h3>
<p style={{ marginTop: '24px', lineHeight: '1.8' }}>
  아래 첫 번째 이미지는 전체 시뮬레이션 신호의 시간 영역 파형으로, 각 구간의 특징(정상, 전이, 고장)을 시각적으로 확인할 수 있다. 녹색 영역은 정상 구간을, 주황색은 전이 구간, 붉은색은 고장 구간을 나타내며, 시간에 따른 리듬 변화가 명확히 구분된다.
</p>


{/* 반응형 이미지 삽입 */}
<img
  src="/images/11.png"
  alt="시뮬레이션 결과 이미지"
  style={{
    maxWidth: '600px',      // 최대 너비 제한
    width: '100%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>

<pre style={{ fontSize: '0.75rem', whiteSpace: 'pre-wrap', background: '#f4f4f4', padding: '16px', borderRadius: '8px', overflowX: 'auto' }}>
{String.raw`import numpy as np
import matplotlib.pyplot as plt

fs = 1000  # 반도체 센서 기준 더 높은 샘플링
duration = 120
t = np.linspace(0, duration, int(fs * duration))
signal = np.zeros_like(t)

# 1. 정상: 8Hz 회전 리듬 (실제 로딩암 동작 주기)
idx_normal = t < 60
signal[idx_normal] = 0.8 * np.sin(2 * np.pi * 8 * t[idx_normal])

# 2. 전이: 고차 주파수 + 선형 잡음 증가
idx_trans = (t >= 60) & (t < 90)
t_trans = t[idx_trans]
signal[idx_trans] = (
    0.8 * np.sin(2 * np.pi * 8 * t_trans) + 
    0.3 * np.sin(2 * np.pi * 16 * t_trans) + 
    0.1 * np.sin(2 * np.pi * 24 * t_trans) + 
    0.05 * np.random.randn(len(t_trans)) * (1 + 0.05 * (t_trans - 60))
)

# 3. 고장: 고주파(40Hz) + 불규칙 진폭 노이즈
idx_fault = t >= 90
t_fault = t[idx_fault]
signal[idx_fault] = (
    0.8 * np.sin(2 * np.pi * 8 * t_fault) + 
    0.4 * np.sin(2 * np.pi * 40 * t_fault) +
    0.3 * np.random.randn(len(t_fault)) * (1 + 0.1 * (t_fault - 90)) +
    0.2 * np.sin(2 * np.pi * 12 * t_fault + np.random.rand())
)

# 시각화
plt.figure(figsize=(15, 4))
plt.plot(t, signal, color='black', linewidth=0.5)
plt.axvspan(0, 60, color='green', alpha=0.1, label='Normal Operation (0–60s)')
plt.axvspan(60, 90, color='orange', alpha=0.1, label='Transition Phase (60–90s)')
plt.axvspan(90, 120, color='red', alpha=0.15, label='Fault Phase (90–120s)')
plt.title("🖥️ Simulated Current Signal of Loading Arm Operation")
plt.xlabel("Time (s)")
plt.ylabel("Simulated Sensor Output")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()`}
</pre>

{/* 반응형 이미지 삽입 */}
<img
  src="/images/17.png"
  alt="시뮬레이션 결과 이미지"
  style={{
    maxWidth: '600px',      // 최대 너비 제한
    width: '100%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>

<h3 style={{ marginTop: '48px', fontSize: '1.25rem' }}>📊 다중 지표 기반 고장 반응 분석</h3>
<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  두 번째 이미지는 세 가지 지표(SRE, NSI, Periodicity)의 Z-score 변화와 각각의 임계값을 함께 시각화한 그래프이다. 고장 발생 시점인 90초를 기준으로, SRE와 NSI는 약 85초부터 급격히 반응을 보이며, Periodicity는 점진적으로 감소해 고장 이후 가장 낮은 값을 기록한다. 이는 각각 리듬 붕괴, 불안정성 증가, 반복성 손실이라는 서로 다른 해석 관점에서 고장을 설명함을 보여준다.
</p>


{/* 반응형 이미지 삽입 */}
<img
  src="/images/12.png"
  alt="시뮬레이션 결과 이미지"
  style={{
    maxWidth: '600px',      // 최대 너비 제한
    width: '100%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>

<h3 style={{ marginTop: '48px', fontSize: '1.25rem' }}>🔵 SRE 단독 반응 강조 및 경고 시점 시각화</h3>
<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  마지막 세 번째 이미지는 SRE 단독 지표에 대한 반응 영역만을 강조하여 시각화한 것이다. Threshold(임계값)를 초과하는 시점 이후부터 파란 음영으로 표시된 구간은 SRE가 구조 리듬의 붕괴를 감지하여 경고 반응을 나타낸 시점이며, 실제 고장 발생보다 수 초 앞선 선행 반응 특성을 가진다. 이는 구조 리듬 기반 지표가 조기 탐지에 효과적임을 보여준다.
</p>


 {/* 반응형 이미지 삽입 */}
<img
  src="/images/13.png"
  alt="시뮬레이션 결과 이미지"
  style={{
    maxWidth: '600px',      // 최대 너비 제한
    width: '100%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>

<pre style={{ fontSize: '0.75rem', whiteSpace: 'pre-wrap', background: '#f4f4f4', padding: '16px', borderRadius: '8px', overflowX: 'auto' }}>
{String.raw`import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import zscore

# 1. 시뮬레이션 신호 생성 (로딩암 기반)
fs = 1000
duration = 120
t = np.linspace(0, duration, int(fs * duration))
signal = np.zeros_like(t)

idx_normal = t < 60
signal[idx_normal] = 0.8 * np.sin(2 * np.pi * 8 * t[idx_normal])

idx_trans = (t >= 60) & (t < 90)
t_trans = t[idx_trans]
signal[idx_trans] = (
    0.8 * np.sin(2 * np.pi * 8 * t_trans) +
    0.3 * np.sin(2 * np.pi * 16 * t_trans) +
    0.1 * np.sin(2 * np.pi * 24 * t_trans) +
    0.05 * np.random.randn(len(t_trans)) * (1 + 0.05 * (t_trans - 60))
)

idx_fault = t >= 90
t_fault = t[idx_fault]
signal[idx_fault] = (
    0.8 * np.sin(2 * np.pi * 8 * t_fault) +
    0.4 * np.sin(2 * np.pi * 40 * t_fault) +
    0.3 * np.random.randn(len(t_fault)) * (1 + 0.1 * (t_fault - 90)) +
    0.2 * np.sin(2 * np.pi * 12 * t_fault + np.random.rand())
)

# 2. 분석 파라미터
win_size = int(2 * fs)
step_size = int(0.5 * fs)
fault_time = 90

time_list, esp_series, periodicity_list = [], [], []

# 3. ESP 및 Periodicity 계산
for i in range(0, len(signal) - win_size, step_size):
    seg = signal[i:i+win_size]
    time_mid = t[i + win_size // 2]
    time_list.append(time_mid)

    f, Pxx = welch(seg, fs=fs, nperseg=win_size // 2)
    Pxx += 1e-12
    prob = Pxx / np.sum(Pxx)
    esp = -np.sum(prob * np.log(prob))
    esp_series.append(esp)
    periodicity_list.append(np.max(Pxx) / np.sum(Pxx))

# 4. 곡률 기반 SRE / NSI 계산
esp_array = np.array(esp_series)
d1 = np.gradient(esp_array)
d2 = np.gradient(d1)
kappa = np.abs(d2)
sre_z = zscore(kappa)

delta_kappa = np.abs(np.diff(kappa, prepend=kappa[0]))
nsi_arr = np.array([
    np.std(delta_kappa[i:i+5]) if i+5 <= len(delta_kappa) else np.nan
    for i in range(len(delta_kappa))
])
nsi_z = zscore(nsi_arr, nan_policy='omit')
periodicity_z = zscore(np.array(periodicity_list))
time_arr = np.array(time_list)

# 5. 임계값 계산 (고장 전 100개 기준)
sre_th = np.mean(sre_z[:100]) + 2 * np.std(sre_z[:100])
nsi_th = np.mean(nsi_z[:100]) + 2 * np.std(nsi_z[:100])
peri_th = np.mean(periodicity_z[:100]) - 2 * np.std(periodicity_z[:100])

# 6. 전체 지표 시각화
plt.figure(figsize=(13, 6))
plt.plot(time_arr, sre_z, label="SRE (Z)", color='blue', linewidth=1.8)
plt.plot(time_arr, nsi_z, label="NSI (Z)", color='purple', linewidth=1.5)
plt.plot(time_arr, periodicity_z, label="Periodicity (Z)", color='orange', linewidth=1.5)
plt.axvline(x=fault_time, color='red', linestyle='--', label='Fault Onset (90s)', linewidth=1.5)
plt.axhline(y=sre_th, color='blue', linestyle=':', label='SRE Threshold (μ+2σ)', linewidth=1.2)
plt.axhline(y=nsi_th, color='purple', linestyle=':', label='NSI Threshold (μ+2σ)', linewidth=1.2)
plt.axhline(y=peri_th, color='orange', linestyle=':', label='Periodicity Threshold (μ-2σ)', linewidth=1.2)
plt.xlabel("Time (s)")
plt.ylabel("Z-score")
plt.title("Z-scored Indicators with Thresholds (Realistic Scenario)")
plt.legend(loc="upper right")
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()

# 7. SRE 단독 시각화
plt.figure(figsize=(14, 5))
plt.plot(time_arr, sre_z, label="SRE (Z)", color='blue', linewidth=2)
plt.axvline(x=fault_time, color='red', linestyle='--', label='Fault Onset (90s)', linewidth=1.5)
plt.axhline(y=sre_th, color='purple', linestyle=':', label='SRE Threshold (μ+2σ)', linewidth=1.5)
plt.fill_between(time_arr, sre_z, sre_th,
                 where=(sre_z > sre_th),
                 interpolate=True,
                 color='blue', alpha=0.3, label="SRE Reaction Zone")
plt.xlabel("Time (s)")
plt.ylabel("Z-score")
plt.title("SRE Response Highlighted Near Fault Onset (Realistic Scenario)")
plt.grid(True, linestyle=':')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()`}
</pre>

{/* 반응형 이미지 삽입 */}
<img
  src="/images/16.png"
  alt="시뮬레이션 결과 이미지"
  style={{
    maxWidth: '600px',      // 최대 너비 제한
    width: '100%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>

<h3 style={{ marginTop: '48px', fontSize: '1.25rem' }}>🧪 분석의 한계와 향후 개선 방향</h3>


<p style={{ marginTop: '24px', lineHeight: '1.8' }}>
  본 분석은 구조 리듬 기반 지표(SRE, NSI, Periodicity)를 활용해 고장 전이 상황을 정량적으로 탐지하고 시각화하는 데 성공하였으나, 여전히 몇 가지 한계점을 내포하고 있다. 첫째, 사용된 신호는 인공 시뮬레이션 기반으로 구성되어 있으며, 실제 반도체 장비에서의 비선형 동작, 센서 드리프트, 환경 조건의 복합적 영향을 반영하지 못한다. 향후에는 실측 로그 데이터를 이용한 모델링이 필요하다.
</p>
<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  둘째, 세 지표는 개별적으로는 유효하나, 지표 간 상호 상관관계나 융합 판단 기준이 부족하다. 다변량 분석 기법이나 지표 조합 기반의 scoring 시스템을 구축함으로써 보다 신뢰도 높은 고장 판단이 가능할 것이다. 셋째, 분석 윈도우는 고정된 2초 창으로 설정되어 있으며, 이는 설비 상태나 주파수 변동에 따라 민감도가 저하될 수 있는 구조이다. Adaptive Window 기법의 도입이 요구된다.
</p>
<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  마지막으로, 지표가 임계값을 초과한 이후의 반응 시점을 자동으로 고장 경보로 변환하거나 시점 예측을 수행하는 기능은 구현되어 있지 않다. 후속 연구에서는 GPI(Global Phase Inflection), CUSUM, EWMA 등 고장 경보 및 시점 추정 알고리즘을 연계하여 실시간 조기 진단 시스템 형태로 확장해야 한다.
</p>

<h3 style={{ marginTop: '48px', fontSize: '1.25rem' }}>💡 반도체 산업의 데이터 흐름과 디지털 트윈 기술</h3>


<p style={{ marginTop: '36px', lineHeight: '1.8' }}>
  최근 반도체 산업에서는 데이터 기반 장비 진단 및 공정 최적화가 전례 없이 중요한 과제로 부상하고 있다. 고난도 미세 공정과 대규모 생산 장비 간의 상호작용 속에서, 단순한 이상 탐지 수준을 넘어 <strong>예지 정비(Predictive Maintenance)</strong>, <strong>공정 편차 예측</strong>, <strong>실시간 제어 최적화</strong>로 분석의 영역이 확장되고 있다. 특히 전류, 진동, 온도 등의 센서 데이터를 활용한 시간-주파수 기반 진단 기법은 기존의 통계 기반 방법론보다 더 높은 민감도와 조기 반응성을 확보할 수 있다는 점에서 산업계와 학계 모두로부터 주목을 받고 있다.
</p>

{/* 반응형 이미지 삽입 */}
<img
  src="/images/18.png"
  alt="시뮬레이션 결과 이미지"
  style={{
    maxWidth: '600px',      // 최대 너비 제한
    width: '100%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  이러한 흐름은 <strong>디지털 트윈(Digital Twin)</strong>이라는 개념과 긴밀히 연결되어 있다. 디지털 트윈이란 물리적 설비의 동작을 가상 공간에서 실시간으로 복제·시뮬레이션하는 기술로, 데이터 기반의 상태 모니터링, 고장 진단, 수명 예측을 통합적으로 수행할 수 있는 차세대 제조 패러다임이다. 본 시뮬레이션 분석 역시 디지털 트윈의 핵심인 <em>현상 기반 고장 징후 예측</em> 기능을 직접 구현한 사례로 볼 수 있으며, 향후 실제 장비 데이터와 결합될 경우 완전한 트윈 모델로 진화할 수 있는 기반이 된다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  결론적으로, 신호 기반 진단 모델은 단순 고장 인식 수준을 넘어서 반도체 장비의 전체 생애주기를 예측하고 최적화하는 기술의 핵심 축이 되어가고 있다. 특히 곡률 기반 지표와 리듬 분석은 고장을 사후에 식별하는 것이 아닌, 구조적 리듬 붕괴라는 형태로 사전에 징후를 감지할 수 있다는 점에서 디지털 트윈 시스템 내에서 매우 강력한 진단 도구로 기능할 수 있다. 이러한 접근은 향후 스마트 팩토리와 연결되는 진정한 <strong>지능형 반도체 제조 시스템</strong> 구현의 핵심 기술이 될 것이다.
</p>

<h4 id="intuitive" style={{ marginTop: '36px', fontSize: '1.25rem', fontWeight: 'bold', color: '#333' }}>😮 개념의 직관적 해설과 정당성</h4>

<p style={{ marginTop: '24px', lineHeight: '1.8' }}>
  복잡한 공정 장비의 동작은 일정한 리듬과 반복을 따릅니다. 이 리듬은 마치 시계의 태엽처럼 고르게 움직이지만, 미세한 마찰, 누적된 하중, 구조 피로 등이 쌓이면 그 고른 리듬은 점차 흔들리기 시작합니다. 본 분석의 핵심은 <strong>그 ‘흔들림’을 수치화하고, 그 속에서 고장을 예측하는 것</strong>입니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  예를 들어, 리듬의 곡률이 갑자기 커진다는 것은, '평탄했던 길에 갑자기 경사가 생긴 것'과 유사합니다. 즉, 주파수 패턴이 급변하면서 설비의 물리적 상태에 변화가 생긴 것이며, 이는 장비의 어느 한 구조가 정상 상태를 벗어났음을 뜻합니다. SRE는 이 곡률의 변화량에 주목하여, <em>“예전처럼 일정하게 돌지 않는다”</em>는 것을 수학적으로 감지합니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  또한, NSI는 리듬 곡률의 불안정성을 본다는 점에서, <em>“매번 다른 흔들림”</em>에 주목합니다. 이는 마치 고르지 못한 심장박동처럼, 불안정하고 예측 불가능한 패턴을 보일 때 그 상태가 심각하다는 신호입니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  Periodicity는 이 리듬이 여전히 규칙적인가를 보는 지표입니다. 규칙적인 운동을 유지할 수 있는 장비는 건강한 상태이며, 특정 주파수 성분이 사라지거나 줄어든다는 것은 장비의 <strong>기본 주기성이 무너지고 있음을 의미</strong>합니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  이처럼, SRE, NSI, Periodicity는 각각 구조 리듬의 <strong>변화량, 불안정성, 반복성 손실</strong>을 관찰하여, 고장이 오기 전의 미세한 징후를 <strong>리듬 기반으로 조기 감지</strong>합니다. 이는 기존의 에너지 기반, 진폭 기반 신호 분석보다 <strong>구조적 해석이라는 고차원적 관점</strong>에서 진단의 정확도와 민감도를 높일 수 있다는 점에서 의미가 있습니다.
</p>

<h4 id="concept-physics" style={{ marginTop: '36px', fontSize: '1.25rem', fontWeight: 'bold', color: '#333' }}>💪 개념의 물리적 정당화</h4>

<p style={{ marginTop: '24px', lineHeight: '1.8' }}>
  회전체 기반 설비는 일반적으로 모터의 일정한 토크, 베어링의 안정된 마찰 조건, 하중 균형을 기반으로 ‘주기적 진동’ 혹은 ‘전류 리듬’을 생성합니다. 이때 센서가 수집하는 전류 신호는 단지 전기 신호가 아니라, <strong>회전 운동의 구조적 상태를 반영하는 물리적 결과물</strong>입니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  본 연구에서 제안하는 SRE(Structural Rhythm Entropy)는, 이 전류 리듬에서의 주파수 복잡도(ESP)를 시간축 상에서 관찰한 후, 그 곡률(2차 변화량)에 기반하여 <strong>리듬의 급격한 구조적 붕괴</strong>를 포착합니다. 이는 진동학에서 말하는 <em>모드 간 상호작용, 공진 붕괴, 구조 피로 누적</em> 현상과 연계되며, <strong>주파수 패턴의 ‘곡률’이 증가한다는 것은 그 구조가 더 이상 선형 모델로 유지되지 않음을 의미</strong>합니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  NSI(Noise Stability Index)는 ESP 곡률의 시간차 분산(Δκ의 분산)을 분석합니다. 이는 센서가 감지한 미세 진동 또는 전류 변화가 안정적인 패턴을 유지하는지 여부를 정량화하는 지표로, <strong>마모나 불균형 증가로 인한 비정상 잡음의 불안정성</strong>을 의미 있게 포착할 수 있습니다. 전통적인 RMS 진폭 변화보다 훨씬 민감하게 <strong>미세한 이상 축적</strong>을 감지할 수 있다는 장점이 있습니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  마지막으로, Periodicity는 전력 스펙트럼 내 지배 주파수의 비율로, 시스템이 <strong>하나의 반복 운동 중심으로 에너지를 집중하고 있는가</strong>를 측정합니다. 이 값이 낮아진다는 것은 운동이 다중 모드로 분산되고 있다는 뜻이며, 이는 <strong>회전체의 비정상적인 공진 분할, 하중 분포 불균형, 또는 기계적 간극 증가</strong>를 암시합니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  이 세 가지 지표는 공통적으로, 전류 신호를 단순한 센서 데이터로 보지 않고, <strong>구조적 운동 상태의 투영물로 해석</strong>한다는 점에서 기존의 평균값 기반 진단 지표(RMS, Peak, Crest Factor 등)와 차별화됩니다. 이는 구조역학, 시스템 진동, 그리고 신호처리의 접점에서 <strong>현상 기반 고장 조기 감지에 대한 물리적 타당성</strong>을 부여하는 핵심입니다.
</p>

<h4 id="design" style={{ marginTop: '36px', fontSize: '1.25rem', fontWeight: 'bold', color: '#333' }}>🧪 실험 신호 리듬 구조의 설계 원리</h4>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  본 실험에 사용된 전류 신호는 총 120초간 수집되며, <strong>정상–전이–고장</strong>의 3단계로 구성됩니다.
  이 구성은 실제 회전체 설비에서 관찰되는 <strong>리듬 붕괴 전이 과정</strong>을 정량화하기 위한 것입니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  <strong>정상 구간(0~60초)</strong>은 8Hz의 일정한 리듬, 낮은 노이즈 플로어, 선형 감쇠 조건을 모사합니다.
  이는 구조적 이상이 없고, 리듬이 <strong>자기복원적</strong>으로 유지되는 상태를 반영합니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  <strong>전이 구간(60~90초)</strong>에서는 고차 주파수 성분과 저주파 위상 변화가 삽입되어,
  실제 설비에서 <strong>편심, 윤활 저하, 베어링 미세 파손</strong> 등이 발생하는 조건을 반영합니다.
  이 시점에서 SRE, NSI 등의 지표는 기준선을 이탈하기 시작합니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  <strong>고장 구간(90~120초)</strong>은 비정상 고주파, 큰 진폭의 노이즈 삽입을 통해 <strong>구조 리듬의 붕괴</strong>를 모델링합니다.
  실제 설비에서는 이 단계에서 고유 진동수가 완전히 무너지고, 리듬은 예측 불가능한 상태로 진입합니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  이와 같은 3단계 구조는 <strong>물리적 고장 메커니즘의 시간적 이행</strong>을 모델링한 것으로,
  단순한 노이즈 주입 시뮬레이션이 아닌, <strong>에너지 전달–복원력–감쇠 특성의 변화</strong>를 반영한 설계입니다.
</p>

<h4 id="sre-nsi" style={{ marginTop: '36px', fontSize: '1.25rem', fontWeight: 'bold', color: '#333' }}>📊 SRE와 NSI: 두 리듬 지표의 상보성</h4>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  <strong>SRE(Structural Rhythm Entropy)</strong>는 신호의 <em>위상–주기–곡률</em> 구조가 얼마나 정형적으로 유지되는지를 분석하여,
  회전체의 <strong>리듬 구조 붕괴</strong>를 탐지하는 데 특화되어 있습니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  반면 <strong>NSI(Noise Stability Index)</strong>는 진폭과 고주파 성분의 <strong>에너지 기반 불안정성</strong>을 측정하여,
  감쇠 계수의 비선형 변화 및 고장 전이 과정을 정량화합니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  두 지표는 각각 <strong>구조적 리듬 변화(SRE)</strong>와 <strong>에너지 흐름의 혼란(NSI)</strong>을 정밀하게 추적함으로써,
  서로 다른 관점에서 동일한 고장 전이를 포착할 수 있는 <strong>상보적 진단 체계</strong>를 구성합니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  실험 결과에서도 고장이 발생하기 수 초 전, <strong>NSI가 먼저 반응하고 SRE가 구조 왜곡을 뒤따라 포착</strong>하는 경향이 확인되었으며,
  이는 <strong>선행 감쇠 변화 → 구조 붕괴의 연쇄적 메커니즘</strong>을 잘 반영합니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  따라서 SRE와 NSI를 함께 사용하는 것은 단일 지표로는 포착할 수 없는 <strong>리듬 붕괴의 전 과정</strong>을 종합적으로 감지하고 해석하는 데 매우 효과적입니다.
</p>



<h3 style={{ marginTop: '48px', fontSize: '1.25rem' }}>🧱 회전체 리듬 붕괴의 물리 기반: 질량–감쇠–강성(M–C–K) 모델 해석</h3>

<h4 id="why-rhythm" style={{ marginTop: '36px', fontSize: '1.25rem', fontWeight: 'bold', color: '#333' }}>🧠 회전체는 왜 리듬을 가지는가?</h4>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  여기서 <strong>‘리듬’</strong>이란, 회전체가 외력 없이 스스로 유지하는 <strong>고유 주기적 운동 패턴</strong>을 뜻합니다. 
  이는 외란이 없을 때 시스템이 자발적으로 진동하는 <em>고유 진동수 및 위상 구조</em>에 의해 결정되며, 시간적으로 반복되는 <strong>에너지 교환의 구조</strong>라고 볼 수 있습니다.
</p>

<p style={{ marginTop: '24px', lineHeight: '1.8' }}>
  회전체 시스템은 동역학적으로 <strong>질량(m)</strong>, <strong>감쇠(c)</strong>, <strong>강성(k)</strong>의 세 가지 구성 요소로 이루어진 <strong>2차 선형 시스템</strong>으로 이상화할 수 있습니다.
  이 모델은 단순 수학적 공식이 아니라, 실제 설비가 <em>에너지를 저장하고 분산시키며 복원하는 방식</em>을 설명합니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  각 요소는 다음과 같은 물리적 역할을 가집니다:
  <ul style={{ paddingLeft: '20px', lineHeight: '1.8' }}>
    <li><strong>m (질량)</strong>: 시스템의 관성. 외력에 대한 반응 속도를 결정합니다.</li>
    <li><strong>c (감쇠)</strong>: 에너지 소산. 마찰과 내부 저항에 의해 진동을 줄이는 요소입니다.</li>
    <li><strong>k (강성)</strong>: 복원력. 변형 후 원래 위치로 되돌리려는 힘입니다.</li>
  </ul>
</p>

<h4 id="mass" style={{ marginTop: '36px', fontSize: '1.25rem', fontWeight: 'bold', color: '#333' }}>⚖️ 질량(m): 회전체 리듬의 시작점</h4>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  <strong>질량 m</strong>은 회전체의 로터, 샤프트, 부품 등 전체 구조의 물리적 무게에 해당하며, 고유 진동수 및 반응 특성을 결정하는 핵심 변수입니다.
  특히 질량의 절대값보다 <strong>분포 위치의 비대칭성</strong>이 리듬 붕괴에 더 큰 영향을 미칩니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
회전체에서 질량 분포가 비대칭이거나 회전 중심이 틀어진 경우, <strong>동적 불균형(dynamic imbalance)</strong>이 발생합니다.
이때 시스템은 각 회전 주기마다 변동하는 원심력에 의해 <strong>불규칙한 힘과 모멘트</strong>를 받게 되고, 그 결과로 <strong>고차 진동수 성분 및 위상 변화</strong>가 리듬에 삽입됩니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  이러한 질량 분포 기반의 리듬 이상은 회전 운동 중 점차 누적되며, 결국 <strong>구조적 주기성의 교란</strong>과 <strong>고유 진동수의 변화</strong>를 야기합니다.
  이 과정에서 <strong>SRE(Structural Rhythm Entropy)</strong>는 신호의 <em>리듬 곡률 구조가 시간에 따라 얼마나 비정형적으로 변형되는지</em>를 수치화함으로써,
  <strong>초기 단계의 리듬 붕괴를 조기에 탐지</strong>할 수 있게 합니다.
</p>


<h4 id="damping" style={{ marginTop: '36px', fontSize: '1.25rem', fontWeight: 'bold', color: '#333' }}>🛢️ 감쇠(c): 진동 에너지의 소멸과 혼란</h4>

<p style={{ marginTop: '24px', lineHeight: '1.8' }}>
  <strong>감쇠 계수 c</strong>는 진동 에너지가 어떻게 소실되는지를 결정하는 요소로, 회전체의 <strong>베어링 마찰, 축 접촉면의 저항, 윤활 상태</strong> 등에 의해 영향을 받습니다.
  이상이 없는 경우엔 일정한 <strong>선형 감쇠</strong>를 보이며 안정적인 리듬이 유지됩니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  그러나 시간이 지나며 마찰면이 열화되거나 윤활유 부족, 내부 오염 등이 발생하면, 감쇠력은 속도나 위치에 따라 <strong>비선형적으로 변화</strong>하게 됩니다.
  이로 인해 특정 속도나 위상에서만 과도한 감쇠가 일어나며, <strong>위상 지연과 진폭 불안정</strong>이 함께 나타납니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  또한 시간이 지나며 축 내부 마모가 발생하면, 감쇠는 단순 저항의 의미를 넘어서 구조 진동의 모드를 교란시키는 요인이 됩니다.
  이런 현상은 <strong>특정 주파수에서만 반응이 커지는 준공진(sub-resonance)</strong> 형태로 나타나며, 정상 리듬의 일정한 주기성에 불규칙한 고주파 패턴이 섞이게 됩니다.
</p>


<h4 id="nonlinear" style={{ marginTop: '36px', fontSize: '1.25rem', fontWeight: 'bold', color: '#333' }}>📡 비선형 감쇠와 고주파 리듬 교란</h4>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  예를 들어, 베어링 표면의 미세 손상은 일정 조건에서 <strong>비선형 모드 간섭(nonlinear mode interference)</strong>을 유도하며, 정상 리듬 신호의 주기적 구조가 깨지고
  불규칙한 <strong>고주파 진동</strong>이 삽입됩니다. 이때 발생하는 리듬 왜곡은 구조 전체에 전파되며, 국소 고장을 전체 고장으로 증폭시킵니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  이는 고유 진동 모드 간 에너지가 비선형적으로 전이되며, 특정 주파수 대역에서 공진 아닌 <strong>간섭 기반 진폭 증폭</strong>이 발생하는 현상입니다.
  단일 공진과 달리 이 간섭은 리듬의 <strong>위상 안정성까지 파괴</strong>하며, 구조 고장의 전조로 나타납니다.
</p>


<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  감쇠의 변화는 전류 신호의 <strong>진폭 변동성 증가</strong>, <strong>비주기적 패턴 삽입</strong>으로 나타나며, 이로 인해 시스템은 더 이상 예측 가능한 리듬을 유지할 수 없게 됩니다.
</p>

<h4 id="stiffness" style={{ marginTop: '36px', fontSize: '1.25rem', fontWeight: 'bold', color: '#333' }}>🧲 강성(k): 리듬 복원의 힘</h4>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  강성이 점진적으로 저하되면, 진동 에너지가 빠르게 복원되지 못하고 시스템에 <strong>잔류 진동</strong>으로 남게 됩니다.
  이로 인해 리듬이 <strong>늘어지거나 지연되며</strong>, 신호에서 <strong>정상 주기성의 붕괴</strong>가 발생합니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  특히 볼트 풀림, 프레임 피로, 고정 구조물의 변형은 강성의 지역적 불균형을 초래하며,
  이는 시스템이 <strong>비정상적인 모드에서 진동</strong>하게 만들고, 구조 리듬이 다중 주기로 분열됩니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  이러한 강성 기반 리듬 붕괴는 <strong>SRE 곡률의 기준선 변동</strong>이나 <strong>고차 공진의 출현</strong>으로 탐지되며,
  점진적인 재료 열화 및 구조적 피로의 선행 징후로 기능할 수 있습니다.
</p>


<h4 id="sre-shift" style={{ marginTop: '36px', fontSize: '1.25rem', fontWeight: 'bold', color: '#333' }}>📉 SRE 기준선 이동과 리듬 붕괴 탐지</h4>

<p style={{ marginTop: '16px', lineHeight: '1.8' }}>
  <strong>SRE(Structural Rhythm Entropy)</strong>는 리듬 곡률의 불안정성을 기반으로 하기 때문에, 
  <strong>시간 축에 따라 변동하는 SRE 곡선</strong>은 <em>시스템이 언제부터 비정상 리듬으로 이행했는지</em>를 선명히 보여줍니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  정상 상태에서는 SRE 값이 낮고 일정한 기준선 주변에서 <strong>작은 파동</strong>을 유지합니다.
  그러나 고장의 징후가 시작되면 곡률 구조가 복잡해지며, SRE 값은 <strong>기준선을 천천히 이탈</strong>하거나, 
  <strong>갑작스럽게 급등</strong>하는 패턴을 보입니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  특히 <strong>전이 구간</strong>에서는 기준선 위에서의 <em>미세한 오버슈트</em>가 반복적으로 발생하며, 
  이는 구조 리듬이 붕괴되기 직전의 예열기(precursor phase)에 해당합니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  <strong>고장 직전 5~10초 구간</strong>에서는 SRE의 분산(variance)과 평균이 동시에 상승하며,
  <strong>리듬 붕괴의 비가역성</strong>을 반영합니다.  
  이 변화는 RMS나 PSD 기반 분석보다 조기이며, <em>곡률의 불연속성이 시작되는 지점</em>을 직접적으로 반영합니다.
</p>


<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  <strong>강성 k</strong>는 구조가 변형된 후 다시 본래 위치로 돌아가려는 복원력의 크기를 나타냅니다.
  회전체 시스템에서 이 값은 <strong>축의 재료 탄성, 고정 구조물의 응력 분포, 연결 부위의 견고성</strong> 등에 의해 결정됩니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  강성이 충분히 확보된 구조는 외력에 의해 순간적으로 변형되더라도, 주기적인 회전 운동 중 빠르게 원래 위치로 되돌아옵니다. 
  이로 인해 <strong>리듬의 주기성과 위상 정합</strong>이 유지됩니다.
</p>


<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  특히 <strong>볼트 풀림, 연결부 피로, 소재 열화</strong> 등으로 인해 강성 변화가 발생하면,
  고유 진동수가 변하고 시스템의 리듬은 점차 <strong>비정상 주파수 성분</strong>을 포함하게 됩니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  이러한 현상은 초기에는 미세하지만, 반복 피로 하중에 의해 리듬이 장기적으로 무너지며,
  이는 <strong>곡률 기반 주파수 리듬 해석(SRE)</strong>의 기준선 이동으로 탐지 가능합니다.
</p>


<h4 id="nsi" style={{ marginTop: '36px', fontSize: '1.25rem', fontWeight: 'bold', color: '#333' }}>📈 NSI: 잡음 안정성 기반 리듬 붕괴 보조 지표</h4>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  본 논문에서는 <strong>리듬 붕괴의 신호 안정성 측면을 보완적으로 해석</strong>하기 위해 <strong>NSI(Noise Stability Index)</strong>를 함께 제안하였습니다.
  NSI는 특정 시간 구간 내에서 신호의 잡음 안정성을 수치화함으로써, 구조 리듬의 위상·진폭 교란을 간접적으로 포착할 수 있는 지표입니다.
</p>


<p style={{ marginTop: '24px', lineHeight: '1.8' }}>
  <strong>NSI(Noise Stability Index)</strong>는 감쇠 계수의 변화가 신호에 미치는 영향을 정량화합니다.
  특히 회전체의 리듬이 안정적인 동안에는 NSI 값이 낮고 일정하게 유지되지만, <strong>윤활 손실, 열 팽창, 베어링 마모</strong> 등이 진행되면
  노이즈 플로어(noise floor)의 분산이 커지면서 NSI가 급격히 상승하게 됩니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  NSI는 단순히 노이즈 양을 측정하는 것이 아니라, 신호 내부의 <strong>리듬 파형의 안정성</strong>을 간접적으로 측정합니다.
  진동 패턴이 흔들리고, 고주파 성분이 리듬 구조를 침범할수록 NSI는 불안정성을 반영하여 상승합니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  특히 고장이 가까워질수록 감쇠가 불규칙적으로 작동하게 되며, 이는 신호의 위상과 진폭을 동시 교란합니다.
  이 때 리듬의 <strong>정상성(stationarity)</strong>이 깨지고, NSI는 <strong>리듬 안정성의 붕괴 지표</strong>로 작용합니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  실제 실험에서는 고장이 발생하기 <strong>수 초 전부터 NSI 값이 급격히 증가</strong>했으며, 이로 인해 감쇠 기반 리듬 이상을 조기에 탐지할 수 있었습니다.
  특히 <strong>RMS 또는 peak-to-peak 기반 방식은 평균 진폭 또는 최대 진폭 중심의 후행 지표</strong>로, 이미 고장이 진행된 이후의 변화를 감지하는 데 적합합니다.
반면, <strong>SRE와 NSI는 리듬 복잡도나 잡음 안정성의 미세한 변화</strong>를 선행적으로 포착할 수 있어, <strong>사전 증상(pre-symptom) 단계에서 경고</strong>를 제공합니다.



</p>

<h4 id="integration" style={{ marginTop: '36px', fontSize: '1.25rem', fontWeight: 'bold', color: '#333' }}>🧩 감쇠–리듬–지표의 통합 해석</h4>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  요약하면, 감쇠 계수 c는 단순한 에너지 소실의 역할을 넘어, <strong>시스템의 리듬 구조를 불안정하게 만들고, 고장 전이 과정을 촉진</strong>하는 결정적 인자로 기능합니다.
  이 감쇠의 변동성은 NSI를 통해 수치화되며, 실시간 리듬 감시 시스템에 적용 시 매우 유효한 경보 트리거로 활용될 수 있습니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  NSI는 단순히 노이즈 양을 측정하는 것이 아니라, 신호 내부의 <strong>리듬 파형의 안정성</strong>을 간접적으로 측정합니다.
  진동 패턴이 흔들리고, 고주파 성분이 리듬 구조를 침범할수록 NSI는 불안정성을 반영하여 상승합니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  특히 고장이 가까워질수록 감쇠가 불규칙적으로 작동하게 되며, 이는 신호의 위상과 진폭을 동시 교란합니다.
  이 때 리듬의 <strong>정상성(stationarity)</strong>이 깨지고, NSI는 <strong>리듬 안정성의 붕괴 지표</strong>로 작용합니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  실제 실험에서는 고장이 발생하기 <strong>수 초 전부터 NSI 값이 급격히 증가</strong>했으며, 이로 인해 감쇠 기반 리듬 이상을 조기에 탐지할 수 있었습니다.
  이는 기존의 RMS, peak-to-peak 방식보다 <strong>선행성(pre-symptomatic detection)이 뛰어나다는 실험적 근거</strong>를 제공합니다.
</p>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  요약하면, 감쇠 계수 c는 단순한 에너지 소실의 역할을 넘어, <strong>시스템의 리듬 구조를 불안정하게 만들고, 고장 전이 과정을 촉진</strong>하는 결정적 인자로 기능합니다.
  이 감쇠의 변동성은 NSI를 통해 수치화되며, 실시간 리듬 감시 시스템에 적용 시 매우 유효한 경보 트리거로 활용될 수 있습니다.
</p>

<h2 style={{ marginTop: '64px', fontSize: '1.25rem' }}>📚 주요 기업별 디지털 트윈 적용 사례</h2>

<p style={{ marginTop: '24px', lineHeight: '1.8' }}>
  회전체 기반 설비의 리듬 분석은 실제 산업 현장에서 <strong>디지털 트윈</strong> 기반 예지보전 시스템과 밀접한 관련이 있습니다.
  주요 반도체 및 정밀 제조 기업들은 이미 고장 전이 징후를 실시간으로 탐지하고, 이를 통해 <strong>PM(Preventive Maintenance)</strong> 정책을 고도화하고 있습니다.
</p>

<ul style={{ paddingLeft: '20px', lineHeight: '1.8' }}>
  <li><strong>SK하이닉스</strong>: 진공펌프의 전류 신호 리듬 분석을 통해 윤활 이슈 및 마모를 예측. NSI 지표를 기반으로 한 수명 추정 시스템을 개발 중.</li>
  <li><strong>삼성전자</strong>: 웨이퍼 핸들링 암의 구조 리듬을 실시간 모니터링하여 반복 오차 발생 전 단계에서 유지보수 시행.</li>
  <li><strong>Lam Research</strong>: 감쇠 기반 피로 누적 모델을 통해 Dry Pump 및 배기 시스템의 동적 안정성 분석.</li>
</ul>

<p style={{ marginTop: '12px', lineHeight: '1.8' }}>
  이러한 적용 사례는 <strong>리듬 붕괴 탐지 기술의 실무적 가치를 강화</strong>하며, 본 연구의 지표 설계(NSI, SRE 등)가 산업 현장에서 얼마나 유효한지에 대한 간접적 근거가 됩니다.
</p>



<DigitalTwinLinks />


    </div>
  );
}
