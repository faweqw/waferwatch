// src/components/SimulationAnalysisPage.js
import React from 'react';
import DigitalTwinLinks from './DigitalTwinLinks';

export default function SimulationAnalysisPage() {
  return (
    <div style={{ padding: '24px', fontFamily: 'Hanna11, sans-serif' }}>
      <h2 style={{ fontSize: '1.5rem', marginBottom: '16px' }}>🖥️ 시뮬레이션 분석</h2>
      <p style={{ color: '#555' }}>
        이 페이지는 고장 진단, 주파수 리듬, 고장 전이 분석 등 시뮬레이션 기반 실험 결과를 시각화하거나 해석하는 공간입니다.
      </p>

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


<h2 style={{ marginTop: '24px', fontWeight: 'bold' }}>📎 전체 시뮬레이션 및 분석 코드 전문</h2>
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

<h4 style={{ marginTop: '32px', fontSize: '1.1rem' }}>📚 주요 기업별 디지털 트윈 적용 사례</h4>
<DigitalTwinLinks />


    </div>
  );
}
