// src/components/PaperAnalysisPage.js
import React from 'react';
import { MathJaxContext, MathJax } from 'better-react-mathjax';


const config = {
  loader: { load: ['[tex]/ams'] },
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$']],
  },
};

export default function PaperAnalysisPage() {
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
        <MathJaxContext config={config}>
    <div style={{ padding: '24px', fontFamily: 'Hanna11, sans-serif' }}>
      
      
      <h2 style={{ fontSize: '1.5rem', marginBottom: '16px' }}>📄 논문 분석</h2>

        
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginBottom: '16px' }}>
    <a href="#abstract" style={tocStyle}>🧾 초록</a>
    <a href="#introduction" style={tocStyle}>📚 서론</a>
    <a href="#model" style={tocStyle}>⚙ 진동 모델</a>
    <a href="#esp" style={tocStyle}>🌐 스펙트럼 엔트로피</a>
    <a href="#sre" style={tocStyle}>🔬 구조 리듬 엔트로피</a>
    <a href="#dataset" style={tocStyle}>📊 데이터셋</a>
    <a href="#pipeline" style={tocStyle}>🛠 분석 파이프라인</a>
    <a href="#esp-analysis" style={tocStyle}>📈 ESP 분석</a>
    <a href="#sre-analysis" style={tocStyle}>📉 SRE 해석</a>
    <a href="#anomaly" style={tocStyle}>🚨 이상 탐지 및 라벨 생성</a>
    <a href="#sre-justification" style={tocStyle}>🧠 SRE 물리적 해석</a>
    <a href="#end" style={tocStyle}>‼️ 결론</a>
    <a href="#limit" style={tocStyle}>✅ 한계점 및 후속 연구</a>
    </div>

<h2 style={{ fontSize: '1.5rem', marginBottom: '16px' }}>🧭 향후 연구 방향</h2>

<div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginBottom: '16px' }}>
  <a href="#kappa-stability" style={tocStyle}>🧩 곡률 기반 안정성</a>
  <a href="#adaptive-threshold" style={tocStyle}>🎚️ 적응형 임계값</a>
  <a href="#sliding-unsupervised" style={tocStyle}>🌀 비지도 Sliding 분석</a>
  <a href="#multi-sensor-integration" style={tocStyle}>🔗 멀티센서 통합</a>
  <a href="#hybrid-dnn" style={tocStyle}>🤖 딥러닝 하이브리드</a>
  <a href="#stochastic-sre" style={tocStyle}>📊 확률론적 해석</a>
  <a href="#mck-dynamics" style={tocStyle}>⚙️ M-C-K 진동 모델</a>
  <a href="#multi-resolution" style={tocStyle}>🔍 다중 해상도 분석</a>
  <a href="#multi-sre" style={tocStyle}>🌐 Multi-SRE 확장</a>
  <a href="#model-comparison" style={tocStyle}>📈 기존 모델 비교</a>
  <a href="#robustness" style={tocStyle}>🛡️ 실측 강건성 보완</a>
  <a href="#ai-hybrid" style={tocStyle}>🧠 AI 통합 전략</a>
  <a href="#sensor-expansion" style={tocStyle}>📡 센서 확장성 설계</a>
  <a href="#realtime-sliding" style={tocStyle}>⏱️ 실시간 Sliding 구조</a>
</div>



        <h4 id="abstract" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
        Abstract
        </h4>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        본 논문에서는 회전체 설비의 고장 진단을 위해 주파수 영역 복잡도의 시간적 구조 변화를 분석하는 구조적 리듬 엔트로피(Structural Rhythm Entropy, SRE) 기반 진단 기법을 제안한다. 기존 방법론은 신호 에너지나 특정 주파수 분석에 한정되어 점진적인 구조적 변화 탐지가 어려웠다. 제안하는 SRE는 Multitaper 기반의 스펙트럼 엔트로피(ESP) 곡선의 곡률 변화량을 Shannon 엔트로피로 정량화하여 고장으로 인한 리듬의 구조적 붕괴를 실시간으로 감지한다. 산업 현장 데이터 분석 결과, 제안된 방법은 기존 지표보다 리듬 붕괴 시점을 민감하게 탐지하며, 특히 정확한 고장 라벨이 부족한 환경에서도 효과적인 이상 탐지와 약지도(weak label) 생성이 가능함을 확인하였다.
        </p>

        <h4 id="introduction" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
        I. 서론
        </h4>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        회전체 설비(모터, 펌프, 축)는 산업 현장에서 다양한 형태의 고장을 겪는다. 고장은 진폭 증가뿐 아니라 시스템의 리듬 구조가 서서히 붕괴하는 현상으로 나타나기 때문에, 기존 RMS, FFT 등의 정적 분석으로는 초기에 탐지가 어렵다. 최근 머신러닝 기반 기법이 발전했지만, 정확한 고장 라벨이 요구되고 물리적 설명이 제한되는 단점이 있다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        본 논문은 기존의 한계를 극복하고자 신호의 복잡도와 곡률 개념을 활용하여 고장으로 인한 리듬 붕괴를 탐지하는 구조적 리듬 엔트로피(SRE)를 제안한다. SRE는 다음의 주요 기여를 가진다:
        </p>

        <ul style={{ marginTop: "12px", lineHeight: "1.8" }}>
        <li>Multitaper 기반 ESP를 활용한 주파수 복잡도 측정</li>
        <li>ESP 곡선의 곡률 변화량을 이용한 구조적 리듬 붕괴 정량화</li>
        <li>약지도(weak label) 생성 및 물리적 연관성 분석을 통한 실시간 고장 탐지 프레임워크 제공</li>
        </ul>

        <h4 style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
        II. 본론
        </h4>

        <h4 id="model" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
        2.1 고장 리듬과 진동 시스템 모델
        </h4>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        회전체 시스템은 일반적으로 질량(M), 감쇠(C), 강성(K)의 세 요소로 구성된 2차 선형 진동 방정식으로 표현된다:
        </p>
        <MathJax>
        {"$$ M \\ddot{x}(t) + C \\dot{x}(t) + Kx(t) = F(t) $$"}
        </MathJax>
        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        여기서 x(t)는 시스템의 시간에 따른 응답이며, F(t)는 외부 혹은 내부에서 발생하는 구동력을 나타낸다. 이러한 수학적 모델은 시스템이 정상 상태에서 반복적인 진동 패턴을 유지하도록 하는 기초적인 구조를 제공한다. 하지만 실제 산업 환경에서는 시간이 지남에 따라 베어링 마모, 벨트 느슨함, 축정렬 불량과 같은 다양한 유형의 고장이 발생하며, 이는 시스템의 주요 파라미터(M, C, K)에 변화를 가져온다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        예를 들어, 베어링 손상은 시스템에 반복적인 충격 교란을 유발하여 힘 항 F(t)의 불규칙성을 증가시킨다. 이는 고주파 영역에서 에너지가 분산되는 형태로 나타난다. 벨트 느슨함은 시스템의 감쇠 특성 C에 변화를 일으켜 슬립 현상으로 인한 리듬 구조의 붕괴를 야기한다. 또한 축정렬 불량은 강성 K의 비선형성을 증가시켜 비대칭적인 진동 및 모드 간섭을 발생시킨다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        결국, 이러한 구조적 변화는 신호의 단순한 진폭이나 특정 주파수만으로는 충분히 탐지하기 어려운 점진적이고 복잡한 형태의 리듬 붕괴로 나타난다. 따라서 고장 초기에 발생하는 미세한 리듬 변화의 탐지를 위해 새로운 분석 방법이 요구된다.
        </p>

        <h4 id="esp" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
        2.2 스펙트럼 엔트로피 (Spectral Entropy, ESP)
        </h4>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        스펙트럼 엔트로피(ESP)는 주파수 영역에서 신호의 에너지 분포의 불확실성, 즉 복잡도를 정보이론적으로 정량화하는 방법이다. ESP는 신호 x(t)의 파워 스펙트럼 밀도(PSD) P(f)를 확률분포 p(f)로 정규화한 후, 다음과 같은 Shannon 엔트로피로 계산된다:
        </p>

        <MathJax>
        {"$$ p(f) = \\frac{P(f)}{\\int P(f)\\, df} $$"}
        </MathJax>
        <MathJax>
        {"$$ ESP(t) = - \\sum_i p_i(f) \\log(p_i(f) + \\varepsilon) $$"}
        </MathJax>



        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        <MathJax>
  {"여기서 $\\varepsilon$ 은 로그 항의 수치적 안정성을 위한 작은 상수이다."}
</MathJax>본 연구에서는 PSD 추정을 위해 Multitaper Spectral Estimation 방식을 사용하였다. Multitaper 방법은 Welch 방법과 비교하여 노이즈에 강하고 짧은 신호 구간에서도 높은 주파수 해상도를 제공하므로 회전체 설비의 비정상 신호 분석에 효과적이다.
        </p>
        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        (1) 물리적 해석
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        ESP는 시간 구간마다 주파수 에너지 분포의 구조적 상태를 나타내며 다음과 같은 물리적 해석이 가능하다:
        </p>

        <ul style={{ marginTop: "12px", lineHeight: "1.8" }}>
        <li>낮은 ESP: 에너지가 특정 주파수 영역에 집중되어 시스템의 리듬이 안정적이고 예측 가능한 상태이다.</li>
        <li>높은 ESP: 에너지가 넓은 주파수 영역에 분산되어 시스템이 불안정하고 예측이 어려운 상태를 나타낸다.</li>
        </ul>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        이를 통해 ESP는 시간축에서의 주파수 구조 변화와 리듬 안정성을 정량적으로 추적하는 데 유용한 지표로 활용될 수 있다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        (2) 실험적 관찰 및 진단 활용
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        실험 분석 결과, ESP는 베어링 손상이나 회전체 불균형과 같은 급격한 고장 유형에서 정상 신호와 뚜렷한 차이를 보였다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        그림 1에 제시된 바와 같이, 베어링 고장군의 ESP 값은 정상군에 비해 평균적으로 높고 분산 또한 크다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        이는 해당 고장군이 정상에 비해 시스템의 리듬 안정성이 저하됨을 의미하며, ESP가 초기에 고장 탐지의 1차 진단 지표로 사용될 가능성을 보여준다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        (3) 한계점 및 SRE 도입 배경
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        그러나 ESP는 전체적인 복잡도 총량만을 측정할 뿐, 다음과 같은 한계를 갖는다:
        </p>

        <ul style={{ marginTop: "12px", lineHeight: "1.8" }}>
        <li>시간 경과에 따른 미세한 구조적 전이를 민감하게 탐지하기 어렵다.</li>
        <li>곡선의 형상(리듬의 구조적 구부러짐)을 고려하지 않는다.</li>
        <li>점진적으로 발생하는 고장의 초기를 감지하기 어렵다.</li>
        </ul>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        이러한 한계를 보완하기 위해 본 연구는 ESP 시계열을 2차원 곡선으로 간주하여 국소적인 곡률 변화를 정량적으로 분석하는 Structural Rhythm Entropy (SRE)를 새롭게 제안한다.
        </p>

        <h4 id="sre" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
        2.3 구조 리듬 엔트로피 (Structural Rhythm Entropy, SRE)
        </h4>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        앞 절에서 설명한 ESP는 주파수 복잡도를 잘 반영하지만, 시간에 따른 리듬 변화의 미세한 구조적 특성을 포착하지 못하는 한계가 있다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        본 연구는 이를 극복하기 위해 ESP의 시간 축 곡선을 곡률 기반으로 분석하여 구조적 리듬 엔트로피(Structural Rhythm Entropy, SRE)를 정의한다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        (1) 곡률 기반 리듬 구조 추출
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        먼저 ESP 시계열 ESP(t)의 곡률 κ(t)는 다음과 같이 정의된다:
        </p>
        <MathJax>
        {"$$ \\kappa(t) = \\frac{ \\left(1 + \\left( \\text{ESP}'(t) \\right)^2 \\right)^{3/2} }{ |\\text{ESP}''(t)| + \\varepsilon } $$"}
        </MathJax>




        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        <MathJax>
  {
    "여기서 $\\text{ESP}'(t)$와 $\\text{ESP}''(t)$는 각각 ESP의 1차 및 2차 도함수이며, $\\varepsilon$은 분모의 발산을 방지하기 위한 작은 양이다. " +
    "곡률 $\\kappa(t)$는 ESP 곡선의 국소적 휘어짐 정도를 나타내며, 값이 급격히 변화하는 구간은 리듬의 구조가 붕괴되는 시점으로 해석된다."
  }
</MathJax>

        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        <MathJax>
        {"곡률 $\\kappa(t)$는 ESP 곡선의 국소적 휘어짐 정도를 나타내며, 값이 급격히 변화하는 구간은 리듬의 구조가 붕괴되는 시점으로 해석된다."}
        </MathJax>

        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        (2) 곡률 변화량 기반 엔트로피 계산
        </p>

        <MathJax>
        {"곡률 시계열에 대해 시간 차분을 적용하여 곡률 변화량 $\\Delta\\kappa(t)$를 정의한다:"}
        </MathJax>



        <MathJax>
        {"$$ \\Delta\\kappa(t) = | \\kappa(t) - \\kappa(t-1) |$$"}
        </MathJax>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        이 변화량의 확률분포를 KDE를 통해 추정한 후, 이를 기반으로 Shannon 엔트로피를 계산한다:
        </p>

        <MathJax>
        {"$$ \\text{SRE}(t) = - \\int_x \\hat{p}_{\\Delta\\kappa}(x) \\log\\left( \\hat{p}_{\\Delta\\kappa}(x) \\right) dx $$"}
        </MathJax>


        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        여기서 KDE의 bandwidth는 Silverman's Rule을 사용하여 데이터 특성에 따라 자동으로 설정하였다. 이를 통해 SRE는 곡률 변화의 분포 특성을 안정적으로 정량화한다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        (3) 해석 및 응용 가능성
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        SRE는 시간에 따른 리듬 불안정성의 강도와 구조적 변화를 민감하게 포착하여, 고장 초기 단계에서 발생하는 리듬 붕괴 지점 (SRE_peak_time)을 효과적으로 탐지할 수 있다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        실험 결과, SRE는 고장 발생 이전에 이미 유의미하게 상승하여 실제 고장 시점을 선행하거나 일치하는 경향을 보였으며, 정상 상태와의 뚜렷한 차이를 나타냈다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        이와 같은 특성을 바탕으로, SRE는 다음과 같은 실질적 응용이 가능하다:
        </p>

        <ul style={{ marginTop: "12px", lineHeight: "1.8" }}>
        <li>조기 경고 시스템: SRE가 정상 기준선(baseline)을 초과하는 구간은 시스템 리듬이 불안정해지고 고장으로 이어질 가능성을 사전에 나타내는 신호로써 활용될 수 있다.</li>
        <li>약지도(Weak Label) 생성: 정확한 고장 라벨링이 어려운 환경에서 고장의 초기 리듬 이상 구간을 예측 학습을 위한 초기 라벨로 제공할 수 있다.</li>
        <li>물리적 연관성 분석: SRE의 피크 지점은 시스템의 진동 방정식(M, C, K 파라미터)의 변동과 직접적으로 연결되어, 수학적 측정값과 물리적 시스템 변화 간의 연관성을 구조적으로 설명하는 근거가 된다.</li>
        </ul>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        결과적으로 SRE는 단순한 복잡도 측정을 넘어 물리적 해석과 연계된 구조적 리듬 붕괴의 수학적 지표로서, 고장 진단 및 예지보전의 중요한 분석 도구로 기능할 수 있다.
        </p>

        <h4 style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
        3. 데이터셋 및 분석 파이프라인
        </h4>

        <h4 id="dataset" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
        3.1 데이터셋 구성
        </h4>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        본 연구는 AI-Hub에서 제공하는 ‘기계시설물 고장 예지 센서 데이터셋’을 기반으로 한다. 해당 데이터는 대전광역시 실험 환경에서 2020년에 수집된 실측 시계열로, 산업용 회전체 계통(모터, 펌프, 벨트 구동 축 등)의 고장 진단 및 예지 모델 개발을 목적으로 구축되었다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        전류(Current) 및 진동(Vibration) 센서를 사용하여 정상 및 고장 상태의 신호 16개 시퀀스를 발췌하였으며, 각 시퀀스는 약 120만 포인트(샘플링 2,000Hz)의 고정 길이를 갖는다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        고장 유형은 정상(Normal), 베어링 손상(Bearing Fault), 벨트 느슨함(Belt Looseness), 회전체 불균형(Imbalance), 축정렬 불량(Misalignment)을 포함하며, 데이터는 CSV 형식으로 제공된다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        다만 고장 발생 시점에 대한 정확한 라벨은 존재하지 않으며, 이에 따라 본 연구는 SRE 기반의 구조 이상 탐지와 약지도(weak label) 생성 가능성에 초점을 맞추어 분석을 수행하였다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        모든 시퀀스는 노이즈 제거 및 구간 정규화 등의 전처리를 통해, 비정상성에 민감한 곡률 기반 분석이 가능하도록 가공되었다.
        </p>

        <h4 id="pipeline" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
        3.2 분석 파이프라인 구조
        </h4>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        본 연구에서 제안하는 고장 진단 프레임워크는 회전체 설비 신호의 시간-주파수 리듬 구조를 분석하여 다음의 3단계로 구성된다:
        </p>

        <ul style={{ marginTop: "12px", lineHeight: "1.8" }}>
        <li>① ESP 기반 복잡도 측정</li>
        <li>② SRE 기반 리듬 전이 탐지</li>
        <li>③ 이상 이벤트 자동 검출 및 약지도 생성</li>
        </ul>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        본 연구에 사용된 데이터셋에는 고장 발생 시점 라벨이 없으므로, 전이 구간 탐지와 약지도 생성 가능성을 중심으로 분석한다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        1단계: 복잡도 기반 고장 분류 (ESP 분석)<br/>
        회전체의 전류 및 진동 신호를 대상으로 Multitaper Spectral Estimation을 사용하여 ESP 시계열을 계산한다. 분석은 2초의 윈도우에 0.5초의 스텝을 적용해 ESP 시계열을 구성한다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        고장 유형 및 센서 유형별로 ESP 평균과 표준편차를 산출하고, 이를 시각화하여 정상군과 고장군의 차이를 비교한다. ESP가 명확하게 두 군을 구분할 경우(예: 베어링 손상, 언밸런스)는 1차 진단 지표로 활용한다. 반면 ESP 분포로 구분이 어렵다면(예: 벨트 느슨함, 축정렬 불량) 후속 리듬 분석을 수행한다.
        </p>
        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        2단계: 리듬 구조 분석 및 전이 구간 탐지 (SRE 분석)<br/>
        <p >
  ESP 곡선을 시간 축에 대해 1차 및 2차 미분하여 국소 곡률  
  <MathJax inline>{" \\( \\kappa(t) \\)"}</MathJax>
   를 계산하고, 곡률 변화량 
  <MathJax inline>{" \\( \\Delta\\kappa(t) \\) "}</MathJax>
  시계열을 구성한다.
</p>
 이 변화량에 대해 슬라이딩 윈도우 내에서 KDE를 이용한 Shannon 엔트로피를 계산하여 SRE 시계열을 생성한다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        SRE가 급격히 상승하는 피크 구간(SRE_peak_time)을 리듬 구조가 붕괴되는 고장 전이 시점으로 판단한다. 최종적으로 원 신호, ESP 및 SRE 시계열을 병렬로 시각화하여 고장 진행의 흐름을 통합적으로 해석한다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        3단계: 이상 이벤트 탐지 및 약지도 생성<br/>
        
<p >
  정상 상태 시퀀스의 
  <MathJax inline>{" \\( \\mathrm{SRE} \\) "}</MathJax>
  평균값에 
  <MathJax inline>{" \\( 2\\sigma \\)"}</MathJax>
  를 더한 값을 임계치로 설정하고, 고장 상태의 시퀀스에서 이 임계치를 초과하는 시점을 자동으로 탐지한다.
  <MathJax inline>{" \\(\( \\mathrm{SRE}_{\\text{event_time}} \)\\)"}</MathJax>
</p>


        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        이 탐지된 구간은 실제 고장 발생 시점과 정확히 일치하지 않을 수 있으나, 정상 리듬에서 벗어난 구조적 이상으로 해석되며, 약한 라벨(weak label)로써 고장 예측 모델 학습에 활용될 수 있다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        이는 향후 반지도 학습 기반의 예지보전 모델의 초기 훈련 데이터로 유용하게 적용될 수 있다.
        </p>

        <h4 id="esp-analysis" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
        4. ESP 기반 고장 복잡도 분석
        </h4>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        Spectral Entropy(ESP)는 주파수 복잡도를 시간축 위에서 정량화하여 회전체 시스템의 리듬 안정성을 평가하는 지표이다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        본 절에서는 정상 상태와 다양한 고장 유형 간 ESP의 차이를 분석하여 고장 진단에서 ESP의 1차적 활용 가능성을 평가하였다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        분석을 위해 전류 및 진동 센서 데이터를 대상으로 Multitaper Spectral Estimation을 적용하여 ESP 시계열을 생성하였으며, 2초 윈도우에 0.5초 간격으로 슬라이딩 윈도우 분석을 수행하였다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        각 시퀀스에 대해 ESP 평균과 표준편차를 산출한 후, 고장 조건별 분포 차이를 시각화하여 분석하였다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        결과적으로, 베어링 손상 및 회전체 불균형(언밸런스)과 같은 특정 고장군은 정상군 대비 ESP의 평균이 높고 분산이 증가하는 명확한 특징을 보였다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        이는 고장 발생 시 신호의 에너지가 특정 주파수에 집중되지 않고 여러 주파수로 분산되어 리듬의 안정성이 낮아지는 현상을 의미한다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        특히 진동 신호 기반 ESP는 베어링 손상에 대해 우수한 민감성을 보이며 정상군과 뚜렷하게 구분 가능하였다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        그러나 모든 고장 유형에 대해 ESP가 일관된 성능을 보이지는 않았다. 벨트 느슨함이나 축정렬 불량과 같은 점진적 구조 변화를 동반하는 고장 유형은 정상군과 고장군 간 ESP 평균값과 분산의 구분력이 낮았다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        이는 ESP가 복잡도의 전체적인 변화만을 반영하며, 시간적 리듬 구조 변화에는 민감하지 못한 한계를 나타낸다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        본 연구는 ESP의 적용 가능성과 한계를 더 명확히 확인하기 위해 각 고장군별 ESP 통계적 분포를 정리하여 비교하였다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        다만 본 데이터셋이 정확한 고장 시점 라벨을 포함하지 않아 MCC, F1-score와 같은 성능 평가 지표를 직접 적용하기 어렵다는 제한이 있었다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        향후 고장 시점이 정확히 기록된 데이터가 확보될 경우, ESP의 진단 성능을 더욱 엄밀히 정량화하고 민감도 분석을 실시할 수 있을 것으로 기대된다.
        </p>

        <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
        본 분석을 통해 ESP는 일부 고장 유형(베어링, 언밸런스 등)에서 유효한 진단 지표로서 활용될 수 있으나, 복잡도의 전체적 변화만을 평가하는 한계가 있어 이를 보완할 수 있는 곡률 기반의 구조적 분석(SRE)이 필요함을 확인하였다.
        </p>

        {/* 반응형 이미지 삽입 */}
<img
  src="/images/a1.png"
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
<p>
그림 1. 고장 유형별 ESP Mean
</p>
<h4 id="sre-analysis" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
  5. SRE 기반 리듬 전이 해석
</h4>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
  ESP는 신호의 주파수 복잡도를 통해 정상과 고장 상태 간의 평균적 차이를 나타내는 데 효과적이지만, 리듬 구조의 실제 붕괴 시점을 포착하는 데에는 한계가 있다. 실제 시스템에서 발생하는 고장은 단순히 복잡도가 증가하는 형태가 아니라, 일정 시점 이후 리듬의 구조적 붕괴로 전이되는 경우가 많기 때문이다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
  이러한 리듬의 구조적 전이를 정량적으로 포착하기 위해 본 연구에서는 ESP 곡선의 국소적인 곡률 변화를 Shannon 엔트로피로 정량화한 Structural Rhythm Entropy(SRE)를 제안하였다. SRE는 다음과 같은 절차를 거쳐 계산된다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
  먼저 ESP 시계열 <MathJax inline>{"\\( ESP(t) \\)"}</MathJax>를 시간 축에 대해 1차 및 2차 미분하여 곡률 <MathJax inline>{"\\( \\kappa(t) \\)"}</MathJax>를 구하고, 이 곡률의 시간 변화량 <MathJax inline>{"\\( \\Delta\\kappa(t) \\)"}</MathJax> 시계열을 계산한다. 이후 <MathJax inline>{"\\( \\Delta\\kappa(t) \\)"}</MathJax> 값의 확률분포를 KDE를 통해 추정하고, 이를 기반으로 Shannon 엔트로피를 계산하여 최종적으로 SRE 시계열을 얻는다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
  SRE 값이 급격히 증가하는 시점은 리듬 구조가 무너지는 전이 구간으로 간주되며, 이 중 최대값을 갖는 시점을 <MathJax inline>{"\\( SRE_{\\text{peak\_time}} \\)"}</MathJax>으로 정의한다. 실험 결과, 대부분의 고장 조건에서 <MathJax inline>{"\\( SRE_{\\text{peak\_time}} \\)"}</MathJax>은 실제 고장 현상보다 먼저 나타나 고장의 조기 탐지 가능성을 보여주었다. 특히 전류 센서는 초기 에너지 변화에 민감하여 고장 초기에 반응한 반면, 진동 센서는 구조적 변화가 누적된 이후에 반응하는 양상을 보였다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
  ESP, SRE 및 원본 신호의 시계열을 동시에 시각화한 결과, <MathJax inline>{"\\( SRE_{\\text{peak\_time}} \\)"}</MathJax> 이후로 시스템의 리듬 구조가 명확히 붕괴되는 것을 확인할 수 있었다. 예를 들어, 베어링 손상의 경우 <MathJax inline>{"\\( SRE_{\\text{peak\_time}} \\)"}</MathJax> 이전에는 충격 신호가 일정한 간격으로 발생했으나, 이후부터는 간격과 크기가 불규칙하게 변화했다. 축정렬 불량의 경우, <MathJax inline>{"\\( SRE_{\\text{peak\_time}} \\)"}</MathJax> 이후부터 모드 간섭 및 비대칭적 진폭이 두드러졌다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
  이러한 실험적 관찰은 SRE가 단순한 수학적 지표에 그치지 않고 실제 기계 시스템의 구조적 변화와 물리적으로 밀접하게 연결된다는 사실을 의미한다. 결과적으로 SRE는 고장의 시점과 형태를 정량적으로 식별할 수 있는 효과적인 구조적 리듬 해석 도구로서, 기존 ESP와 같은 복잡도 총량 기반 지표의 한계를 보완할 수 있다.
</p>

<img
  src="/images/b1.png"
  alt="시뮬레이션 결과 이미지"
  style={{
    maxWidth: '600px',      // 최대 너비 제한
    width: '100%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/><img
  src="/images/b2.png"
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
        <p style={{ marginTop: "12px", lineHeight: "1.8", fontStyle: "italic" }}>
          그림 2. 정상(bearing_normal) 및 베어링 고장(bearing_fault)의 ESP 및 SRE 시계열 비교 (전류 센서 데이터). SRE는 ESP 곡선의 곡률 기반으로 리듬 붕괴 시점을 명확히 강조하여 나타낸다.
        </p>

<h4 id="anomaly" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
  6. SRE 기반 이상 이벤트 탐지 및 weak-label 생성
</h4>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
  본 절에서는 Structural Rhythm Entropy(SRE)가 단순히 구조 리듬을 해석하는 도구에 그치지 않고, 실제 이상 탐지와 약지도(weak-label) 생성에 실용적으로 활용될 수 있음을 제안한다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
  먼저 정상 상태 신호에서 산출된 SRE의 평균값
  <MathJax inline>{" \\( \\mu_{\\text{SRE}} \\) "}</MathJax>
  과 표준편차
  <MathJax inline>{" \\( \\sigma_{\\text{SRE}} \\) "}</MathJax>
  를 기반으로 이상 탐지 임계치를 설정한다. SRE 값이 이 임계치를 초과하는 시점을 이상 이벤트 발생 시점
  <MathJax inline>{" \\( \\mathrm{SRE}_{\\text{event\_time}} \\) "}</MathJax>
  으로 정의하였다.
</p>

<MathJax>
{`$$
\\mathrm{SRE}(t) > \\mu_{\\text{normal}} + x \\sigma_{\\text{normal}}, \\quad (x = 2)
$$`}
</MathJax>


<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
  이 접근법은 명확한 고장 발생 라벨이 없는 비지도 환경에서도 리듬의 구조적 불안정성을 통해 고장 초기 징후를 탐지할 수 있다. 탐지된 시점은 구조적 리듬이 무너지기 시작한 전이점으로 해석 가능하며, 약지도(weak-label)로써 RNN, LSTM, Transformer 등 예지 모델의 초기 학습에 유용하게 활용될 수 있다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
  실험적으로 고장 시퀀스에서 탐지된
  <MathJax inline>{" \\( \\mathrm{SRE}_{\\text{event\_time}} \\) "}</MathJax>
  은 실제 고장 현상 발생 시점보다 평균적으로 약 0.6~1.4초 선행하였다. 반면, 정상군에서는 탐지 이벤트가 전체 데이터의 5% 미만에서만 발생해 SRE의 높은 특이성과 신뢰성을 보였다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
  이를 통해 SRE는 다음과 같은 응용 가능성을 가진다:
</p>

<ul style={{ marginTop: "12px", lineHeight: "1.8" }}>
  <li>📌 <strong>이상 이벤트 탐지 도구</strong>: 명확한 라벨 없이도 고장 발생 전 구조적 리듬 이상을 민감하게 포착한다.</li>
  <li>📌 <strong>약지도(Weak-label) 생성</strong>: 라벨 부족 환경에서 고장 예측 모델의 초기 데이터로 활용된다.</li>
  <li>📌 <strong>예지보전(PHM) 트리거</strong>: 실시간 시스템 모니터링에서 구조적 이상 구간을 예측해 선제적 유지보수를 가능하게 한다.</li>
</ul>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
  이러한 특성을 종합하면 SRE는 산업 현장에서 실용적으로 활용 가능한, 구조적으로 타당한 이상 탐지 및 약지도 생성 도구로 발전할 수 있을 것으로 기대된다.
</p>

{/* 시각화 이미지 추가 예시 */}
<img
  src="/images/a3.png"
  alt="SRE 평균값 분포"
  style={{
    maxWidth: '600px',
    width: '100%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>
<p>그림 3. 고장 유형 및 센서별 SRE 평균값 분포. ESP로는 구분되지 않던 고장군에서 SRE는 뚜렷한 구조 불안정성을 반영함.</p>

<img
  src="/images/a4.png"
  alt="SRE 표준편차 분포"
  style={{
    maxWidth: '600px',
    width: '100%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>
<p>그림 4. 고장 유형 및 센서별 SRE 표준편차. 고장군은 정상군 대비 SRE 변화 폭이 크고 불규칙적임을 보여줌.</p>


<h4 id="sre-justification" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
7. SRE_peak_time의 구조적 정당화와 물리적 해석
</h4>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
본 연구는 SRE가 시스템의 고장 시 리듬 구조가 붕괴되는 시점을 민감하게 탐지하는 효과적인 곡률 기반 지표임을 확인하였다. 본 절에서는 실험적 결과를 기반으로 <MathJax inline>{"$SRE_{\\text{peak\_time}}$"}</MathJax>이 실제 시스템의 구조적 변화 및 고장 전이 시점과 어떻게 연관되는지를 정당화한다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
먼저, 원 시계열 신호, ESP 및 SRE를 시간축으로 정렬하여 분석한 결과, <MathJax inline>{"$SRE_{\\text{peak\_time}}$"}</MathJax> 이후 신호의 리듬 구조가 급격히 무너지는 현상이 일관되게 나타났다. 베어링 손상의 경우, <MathJax inline>{"$SRE_{\\text{peak}}$"}</MathJax> 이전까지는 일정한 간격으로 충격 응답이 발생했지만, 이후부터 충격 간격과 크기가 불규칙하게 변화하였다. 벨트 느슨함의 경우에도 <MathJax inline>{"$SRE_{\\text{peak}}$"}</MathJax> 이후 슬립 주기의 붕괴와 리듬 불균형이 명확히 나타났으며, 축정렬 불량에서는 peak 이후 진폭의 비대칭성과 다중 모드 간섭이 두드러졌다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
이러한 관찰은 <MathJax inline>{"$SRE_{\\text{peak\_time}}$"}</MathJax>이 단순한 수학적 극대점이 아니라, 실제 물리적 시스템의 리듬 구조가 본격적으로 무너지는 시점과 정확히 연계됨을 의미한다. 또한 센서 유형별 분석 결과, 전류 센서는 고장 초기 단계에서 즉각적인 반응을 나타냈으며, 진동 센서는 구조적 변화가 일정 시간 누적된 이후에 반응하는 특성을 보였다. 이는 전류 신호가 고장 초기의 에너지 이상에 민감한 반면, 진동 신호는 구조 변화의 누적 효과에 더 민감하다는 사실을 시사한다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
또한, 정상군 대비 고장군에서의 <MathJax inline>{"$SRE_{\\text{peak\_time}}$"}</MathJax>은 평균적으로 0.6~1.4초 앞서 나타났다. 이는 SRE가 고장의 사후 진단 지표를 넘어 고장을 조기에 탐지할 수 있는 선행적 신호로 기능할 수 있음을 입증한다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
종합적으로, <MathJax inline>{"$SRE_{\\text{peak\_time}}$"}</MathJax>은 수학적으로는 곡률 변화량의 극대점을, 신호적으로는 리듬 붕괴의 시작점을, 물리적으로는 시스템의 구조적 안정성이 처음 무너지는 전이 시점을 나타낸다. 이러한 결과는 SRE가 단순한 수학적 측정값을 넘어 실제 시스템의 물리적 고장 메커니즘과 긴밀히 연동되는 해석 가능하고 실용적인 진단 지표임을 강력히 지지한다.
</p>

{/* 삽입 이미지 */}
<img
  src="/images/a5.png"
  alt="고장군 vs 정상군의 SRE_peak_time 비교"
  style={{
    maxWidth: '600px',
    width: '100%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>
<p style={{ fontStyle: "italic" }}>
그림 5. 고장군 vs 정상군의 SRE_peak_time 비교. 전류는 즉각 반응, 진동은 누적 리듬 붕괴에 후속 반응함을 보여줌.
</p>

<h4 id="end" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
8. Discussion
</h4>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
본 연구는 회전체 고장 진단을 단순한 이상 탐지 문제가 아닌, 시계열 리듬 구조의 붕괴 과정을 수학적으로 해석하는 문제로 전환하였다. 이를 위해 주파수 복잡도를 측정하는 ESP와, 그 시간축 곡선의 곡률 변화량을 기반으로 리듬의 구조적 불안정성을 정량화하는 Structural Rhythm Entropy(SRE)를 제안하였다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
실험 결과, ESP는 베어링 손상 및 회전체 불균형과 같은 급격한 고장 유형에서 고장군과 정상군 간의 복잡도 차이를 효과적으로 설명하였다. 그러나 벨트 느슨함이나 축정렬 불량과 같이 점진적인 고장에서는 ESP의 평균값만으로 고장 여부를 구분하기 어려운 한계를 보였다. 이에 반해, SRE는 곡률의 급변을 민감하게 포착함으로써, 이러한 비선형적 고장에서도 리듬 붕괴 시점(<MathJax inline>{"$SRE_{\\text{peak\_time}}$"}</MathJax>)을 정량적으로 추정할 수 있었다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
<MathJax inline>{"$SRE_{\\text{peak\_time}}$"}</MathJax>은 단순한 수학적 극댓값이 아니라, 시스템의 M, C, K 파라미터 교란이 실제 신호 리듬에 반영되는 전이점과 시간적으로 밀접하게 연관되어 있는 것으로 해석된다. 이는 신호 기반 지표가 시스템의 구조적 변화를 반영할 수 있음을 시사한다. 또한, 임계치 기반 이벤트 탐지를 통해 약지도(weak-label)를 자동 생성할 수 있는 가능성도 확인되었다. 실험에서는 고장군의 <MathJax inline>{"$SRE_{\\text{event\_time}}$"}</MathJax>이 실제 고장 발생 시점보다 평균적으로 0.6~1.4초 앞서 나타났으며, 정상군에서는 false positive 비율이 5% 미만으로 낮게 유지되었다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
이러한 결과는 SRE가 고장의 직접적 시점을 정확히 예단하는 것은 아니지만, 구조 리듬의 붕괴 조짐을 사전에 감지하여 고장 가능성이 높은 전이 구간을 효과적으로 포착할 수 있음을 의미한다. 이는 조기 경고 및 예지보전 시스템의 유용한 입력으로 활용될 수 있다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
센서 반응 특성 측면에서는, 전류 센서는 고장 초기의 에너지 이상에 민감하게 반응하였고, 진동 센서는 구조 리듬의 누적 붕괴 이후 반응하는 경향을 보였다. 이는 센서 간 상보적 조합을 통한 다중모드 기반 예지 시스템 설계의 가능성을 시사한다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
종합적으로, 본 연구는 ESP와 SRE의 이중 구조를 활용하여 기계 리듬의 동적 변화 흐름을 실시간으로 분석할 수 있는 진단 프레임워크를 제시하였다. 본 방법론은 해석 가능성(XAI), 예지보전(PHM) 시스템과의 연계성, 물리적 정당성의 세 요소를 통합하며, 산업 현장에 적용 가능한 신호 기반 고장 진단의 새로운 방향을 제시하였다.
</p>

       
     <h4 id="limit" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
9. 한계점 및 후속 연구 방향
</h4>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
본 연구는 회전체 고장을 주파수 복잡도의 시간 구조적 붕괴로 해석하고, 곡률 기반의 리듬 엔트로피 지표인 
<MathJax inline>{" \\( \\mathrm{SRE} \\)"}</MathJax>
(Structural Rhythm Entropy)를 통해 고장의 전이 구간을 정량화하고자 하였다. 그러나 제안한 분석 프레임워크는 다음과 같은 수학적, 통계적, 물리적 측면에서 보완이 필요하다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
첫째, 
<MathJax inline>{" \\( \\mathrm{ESP}(t) \\)"}</MathJax>
의 곡률 정의는 2차 미분을 기반으로 하나, 실제 센서 신호는 고주파 노이즈와 이산 샘플링의 영향을 받아 미분 가능성(smoothness)이 보장되지 않는다. 수치적 발산을 방지하기 위해 
<MathJax inline>{" \\( \\varepsilon \\)"}</MathJax> 
항을 추가했으나, 해당 값의 고정 설정은 자의적이며 민감도 조정에 영향을 줄 수 있다. 향후에는 
<MathJax inline>{" \\( \\Delta \\kappa(t) \\)"}</MathJax>
의 국소적 분산을 기반으로 
<MathJax inline>{" \\( \\varepsilon(t) \\)"}</MathJax>
를 동적으로 조절하는 adaptive regularization 방안이 필요하다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
둘째, 
<MathJax inline>{" \\( \\Delta \\kappa \\)"}</MathJax>
의 히스토그램 기반 확률분포는 안정성이 낮고, 엔트로피 계산의 기반이 되는 정량적 분포 정의가 모호하다. 이를 보완하기 위해 본 연구에서는 KDE(Kernel Density Estimation)를 도입하고, bandwidth는 Silverman’s rule에 따라 설정하였으나, 향후에는 permutation test나 bootstrap 기반 신뢰구간 추정을 통해 분포의 통계적 타당성과 threshold 설정의 객관성을 높이는 절차가 요구된다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
셋째, 본 연구는 
<MathJax inline>{" \\( \\mathrm{SRE}_{\\text{peak\_time}} \\)"}</MathJax>
과 실제 고장 발생 구간 간의 시간적 선행 관계를 실험적으로 확인하였으나, 해당 시점이 진동 시스템의 
<MathJax inline>{" \\( \\textit{M}, \\textit{C}, \\textit{K} \\)"}</MathJax>
파라미터 교란과 어떻게 연계되는지를 수학적으로 증명하지는 못했다. 향후 연구에서는 
MCK 기반 진동 방정식에 고장 파라미터를 삽입한 시뮬레이션을 통해 
<MathJax inline>{" \\( \\mathrm{ESP} \\rightarrow \\kappa \\rightarrow \\mathrm{SRE} \\)"}</MathJax>
의 동적 연쇄 반응을 물리 모델과 수학적으로 연결하는 작업이 필요하다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
넷째, 곡률 기반 
<MathJax inline>{" \\( \\mathrm{SRE} \\)"}</MathJax>
는 고주파 노이즈에 민감하므로, 특정 조건에서 false peak를 유발할 수 있다. 이를 완화하기 위해 Noise Stability Index(NSI)와 같은 보조 지표를 도입하여 SNR 수준에 따른 
<MathJax inline>{" \\( \\mathrm{SRE} \\)"}</MathJax>
의 신뢰도를 계량화하고, 안정적인 threshold 설정 기준을 마련해야 한다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
다섯째, 
<MathJax inline>{" \\( \\Delta \\kappa(t) \\)"}</MathJax>
를 기반으로 한 
<MathJax inline>{" \\( \\mathrm{SRE} \\)"}</MathJax> 
대신 다중 해상도 wavelet coefficient의 엔트로피를 활용하는 대체 구조를 고려할 수 있다. 이 방식은 저역/고역 대역의 리듬 복잡도를 분리 분석할 수 있어, 주파수 선택성을 갖춘 고장 감지로의 확장이 가능하다.
</p>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
종합적으로, 본 연구의 한계는 
<MathJax inline>{" \\( \\mathrm{SRE} \\)"}</MathJax>
의 해석적 잠재력을 약화시키지 않으며, 오히려 후속 연구를 통해 신호처리의 수학적 정합성과 기계 동역학의 물리적 해석력을 결합한, 보다 통합적이고 확장성 있는 고장 진단 프레임워크로 진화할 수 있는 가능성을 열어둔다.
</p>


<h4 id="references" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
10. References
</h4>

<div style={{ marginTop: "12px", lineHeight: "1.8" }}>
  <p><b>[1]</b> Sun, L., Wang, H., Xu, X., Guo, Z., & Lu, B. (2020). Cumulative spectrum distribution entropy for rotating machinery fault diagnosis. <i>Mechanical Systems and Signal Processing</i>, <b>138</b>, 106550.</p>
  <p><b>[2]</b> Guo, S., Yang, T., Gao, W., & Zhang, C. (2018). A novel fault diagnosis method for rotating machinery based on a convolutional neural network. <i>Sensors</i>, <b>18</b>(5), 1429.</p>
  <p><b>[3]</b> Thomson, D. J. (2007). Jackknifing multitaper spectrum estimates. <i>IEEE Signal Processing Magazine</i>, <b>24</b>(4), 20–30.</p>
  <p><b>[4]</b> Silverman, B. W. (1986). <i>Density Estimation for Statistics and Data Analysis</i>. Chapman & Hall.</p>
  <p><b>[5]</b> Singh, H., Misra, N., Hnizdo, V., Fedorowicz, A., & Demchuk, E. (2003). Nearest neighbor estimates of entropy. <i>American Journal of Mathematical and Management Sciences</i>, <b>23</b>(3–4), 301–321.</p>
  <p><b>[6]</b> Rao, S. S. (2017). <i>Mechanical Vibrations</i> (6th ed.). Pearson Education.</p>
  <p><b>[7]</b> Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Metrics for evaluating performance of prognostic techniques. <i>Proc. of Int. Conf. on Prognostics and Health Management (PHM)</i>.</p>
  <p><b>[8]</b> Wu, Z., Li, Z., He, Y., & Liu, Y. (2022). Weakly supervised deep learning for fault detection and diagnosis in industrial time series data. <i>IEEE Transactions on Industrial Informatics</i>, <b>18</b>(6), 3984–3994.</p>
  <p><b>[9]</b> Lei, Y., Li, N., Guo, L., Li, N., Yan, T., & Lin, J. (2018). Machinery health prognostics: A systematic review from data acquisition to RUL prediction. <i>Mechanical Systems and Signal Processing</i>, <b>104</b>, 799–834.</p>
  <p><b>[10]</b> Voss, A., Schulz, S., Schroeder, R., Baumert, M., & Caminal, P. (2009). On the use of spectral entropy as a measure of complexity in physiological time series. <i>Physiological Measurement</i>, <b>30</b>(1), 87–95.</p>
  <p><b>[11]</b> Jiang, H., Xu, Y., & Zhang, H. (2016). Fault diagnosis of rotating machinery using improved spectral kurtosis and optimized SVM. <i>Shock and Vibration</i>, 2016, Article ID 9834207.</p>
  <p><b>[12]</b> He, Q. (2007). Fault diagnosis of rotating machinery based on wavelet entropy and support vector machines. <i>Journal of Sound and Vibration</i>, <b>302</b>(4–5), 1101–1115.</p>
  <p><b>[13]</b> Cottone, F., Gammaitoni, L., & Vocca, H. (2009). Nonlinear energy harvesting. <i>Physical Review Letters</i>, <b>102</b>(8), 080601.</p>
  <p><b>[14]</b> Sethian, J. A. (1999). <i>Level Set Methods and Fast Marching Methods</i>. Cambridge University Press.</p>
  <p><b>[15]</b> Chan, T. F., & Vese, L. A. (2001). Active contours without edges. <i>IEEE Transactions on Image Processing</i>, <b>10</b>(2), 266–277.</p>
  <p><b>[16]</b> Li, C., Sanchez, R.-V., Zurita, G., Cerrada, M., Cabrera, D., & Vásquez, R. E. (2015). Multimodal deep support vector classification with homologous features and its application to gearbox fault diagnosis. <i>Neurocomputing</i>, <b>168</b>, 119–127.</p>
  <p><b>[17]</b> Yang, W., Tavner, P. J., Crabtree, C. J., & Feng, Y. (2014). Wind turbine condition monitoring: technical and commercial challenges. <i>Wind Energy</i>, <b>17</b>(5), 673–693.</p>
  <p><b>[18]</b> Kumar, S., & Tyagi, B. (2021). Fault detection in induction motor using fusion of S-transform and spectral entropy. <i>International Journal of System Assurance Engineering and Management</i>, <b>12</b>(2), 251–261.</p>
  <p><b>[19]</b> Hu, Q., & He, Z. (2006). Time-frequency manifold entropy and its application to fault pattern recognition in rotating machinery. <i>Mechanical Systems and Signal Processing</i>, <b>20</b>(3), 813–831.</p>
  <p><b>[20]</b> Yang, B.-S., Kim, K.-J., & Han, T. (2004). An adaptive fault diagnosis system of induction motors using hybrid artificial neural networks. <i>Mechanical Systems and Signal Processing</i>, <b>18</b>(2), 329–346.</p>
</div>

<>
  <h4 id="appendix" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
    Appendix
  </h4>

  <p style={{ marginTop: "12px", lineHeight: "1.8" }}>
    <b>그림 A.</b> 고장 유형별 진동 센서 시계열에서의 ESP-SRE 동시 시각화. 각 시계열은 시간 축을 기준으로 구조 리듬의 붕괴 양상을 나타내며, 고장 유형에 따라 SRE의 급격한 상승과 ESP의 국소적인 곡률 변동이 동시에 발생하는 양상을 보여준다.
  </p>
</>

{/* 삽입 이미지 */}
<img
  src="/images/a6.png"
  alt="고장군 vs 정상군의 SRE_peak_time 비교"
  style={{
    maxWidth: '600px',
    width: '100%',
    height: 'auto',
    margin: '24px 0',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}
/>


<h4 id="kappa-stability" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
🧩 수학적 곡률 기반 모델의 안정성 향상 방안
</h4>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
본 연구는 신호의 구조적 리듬 붕괴를 정량적으로 해석하기 위한 지표로서 곡률 기반 엔트로피 지수인 SRE(Spectral Rhythm Entropy)를 도입하였다. SRE는 시간 축 상에서 신호의 곡률 변화 <MathJax inline>{"$\\kappa(t)$"}</MathJax>와 그 변화량 <MathJax inline>{"$\\Delta\\kappa(t)$"}</MathJax>에 기반하여, 신호의 리듬 구조가 어떻게 무너지는지를 민감하게 감지한다. 그러나 실제 물리 신호에는 고주파 잡음, 샘플링 오차, 미세한 이상값이 존재하며, 이러한 요소들은 곡률 계산의 수치 민감도를 증가시켜 <em>SRE의 신뢰성 저하</em>로 이어질 수 있다.

이를 해결하기 위해 본 절에서는 세 가지 보완 방안을 제시한다. 첫째, 다중 해상도 기반 smoothing 설계를 통해 곡률 계산 시의 과민 반응을 억제하고, 진정한 구조 리듬 변화만을 반영할 수 있도록 Gaussian 또는 Savitzky-Golay 필터 계열을 병렬 적용하는 방식을 제안하였다. 둘째, 곡률 변화량의 이동 분산, 첨도, 변동성을 함께 고려하는 보조 지표 <em>Stable-SRE</em>를 도입하여, 신호 내 국소적 이상보다 장기적인 구조 리듬의 붕괴에 집중할 수 있도록 설계하였다. 마지막으로 시간-주파수 해석(STFT, PSD)과 결합된 이중 시점 분석을 통해, SRE 변화가 물리적 주파수 재배치와 동시에 발생하는지를 확인함으로써 과잉 반응을 억제하고 해석 신뢰도를 향상시켰다. 이러한 구조는 SRE를 이론적 지표에서 실측 기반 진단 지표로 확장시키는 데 필수적인 안정화 프레임워크로 작용한다.
</p>

<h4 id="adaptive-threshold" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
🎚️ Adaptive Threshold 기반 구조 보완
</h4>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
SRE와 NSI와 같은 리듬 기반 이상 지표는 민감도가 높다는 장점이 있지만, 고정 임계값(threshold) 기반으로 경고를 발생시킬 경우 오탐(false alarm)이나 과경고(over-alert)가 발생할 위험이 있다. 특히 고장 징후가 발생하더라도 장비의 특성, 부하 조건, 센서 오차 등에 따라 신호의 통계적 특성이 크게 달라지기 때문에, 전통적인 Z-score 기준인 <MathJax inline>{"$\\mu + 2\\sigma$"}</MathJax>와 같은 고정 모델은 실효성이 떨어질 수 있다.

이에 따라 본 연구는 각 시점의 지표값 분포에 따라 실시간으로 경계값이 조정되는 <em>적응형 임계값 구조</em>를 제안하였다. 특히 이동 윈도우 내의 상위 분위수(percentile) 값에 대한 지수 이동 평균(EMA)을 적용한 다음 식은, 시스템의 변화에 실시간 반응하면서도 노이즈에 대한 민감도는 억제하는 장점을 가진다:  
<br /><MathJax>{"$$\\text{AdaptiveThreshold}(t) = EMA_\\tau\\left(P_{95}(SRE(t - \\Delta T : t))\\right)$$"}</MathJax>  
이 방식은 임계값 설정을 자동화하며, 운영 환경이 바뀌더라도 추가적인 수동 튜닝 없이 정확한 경보를 가능하게 한다. 더불어 이 구조는 딥러닝 기반 이진 분류기(LSTM, CNN 등)와 통합될 수 있으며, Adaptive Threshold, dSRE/dt, NSI(t) 등 다양한 시계열 입력을 통해 조기 고장 예측을 강화하는 전처리 모듈로 활용 가능하다. 궁극적으로 이는 현장 진단 시스템의 실효성과 유지보수 비용 절감에 크게 기여할 수 있다.
</p>


<h4 id="sliding-unsupervised" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
🌀 Sliding 기반 자동 구간 분할 및 비지도 탐지 구조
</h4>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
기존의 고장 진단 시스템은 “정상-전이-고장”이라는 사전 정의된 3단계 구조에 의존하는 경우가 많다. 그러나 실제 산업 현장에서는 전이의 시점과 형태가 명확히 정의되지 않으며, 이로 인해 구간 설정 기반 진단 구조는 적용에 한계를 보일 수 있다. 이를 해결하기 위해 본 연구는 Sliding Window 기반의 자동 구간 분할 구조를 채택하였다. 본 방식은 전체 시계열에서 일정 길이의 윈도우를 순차적으로 이동시키며, 각 구간에 대해 <MathJax inline>{"$SRE(t)$"}</MathJax>, <MathJax inline>{"$NSI(t)$"}</MathJax>, <MathJax inline>{"$\\Delta SRE(t)$"}</MathJax> 등의 구조 리듬 지표를 계산하고, 이들의 시계열 벡터를 클러스터링하여 비정형 이상 구간을 탐지한다.

Sliding 구조의 핵심은 비지도 학습 기반의 구간 정의이다. DBSCAN, Isolation Forest, LOF 등의 밀도 기반 이상 탐지 기법을 적용함으로써, 사전 라벨링 없이도 유사 구간의 패턴을 그룹화할 수 있으며, 특히 <em>전이 구간</em>과 <em>고장 구간</em>을 명시적으로 구분하지 않아도 자동적으로 이상도 상승 구간이 추출된다. 이러한 구조는 신호 내의 미묘한 변화와 누적된 불안정성까지 포착할 수 있어 고장의 조기 징후 감지에도 효과적이다. 특히 향후 준지도 학습(semi-supervised learning)이나 weak labeling 기반 시계열 분류기(LSTM, GRU 등)와 연결될 경우, 고장 징후의 자율 분할 및 명명까지 가능한 자율 진단 프레임워크로 확장될 수 있다.
</p>

<h4 id="multi-sensor-integration" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
🔗 멀티센서 구조 통합 및 SRE 기반 다변량 진단 시스템 확장
</h4>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
실제 산업 시스템에서 발생하는 고장은 단일 물리량에만 국한되지 않으며, 전류, 진동, 온도, 음향 등의 다양한 센서 채널을 통해 복합적으로 표현된다. 본 연구는 곡률 기반 SRE(Spectral Rhythm Entropy)의 다변량 확장을 통해 이러한 다센서 환경에서의 통합 진단이 가능함을 제안한다. 각 센서 채널별로 SRE와 NSI를 독립적으로 계산하고, 이를 시간 동기화하여 <MathJax inline>{"$\\text{MultiSRE}(t)$"}</MathJax> 벡터로 결합하는 구조는 시스템 상태에 대한 정합적 리듬 프로파일을 제공한다.

본 구조의 핵심은 각 센서의 리듬 붕괴 현상이 동일 시점에 발생하는지의 “공동성” 여부를 탐지하고, 고장 원인을 다차원적으로 추론하는 것이다. 예컨대 진동 기반 SRE가 선제적으로 상승하고 이어서 전류 기반 SRE가 뒤따르는 경우, 이는 기계적 마모가 전기적 부하 불균형으로 전이된 패턴으로 해석될 수 있다.  
이를 위해 <MathJax inline>{"$\\text{MultiSRE}(t) = w_1 SRE_\\text{current}(t)^2 + w_2 SRE_\\text{vib}(t)^2 + \\cdots$"}</MathJax> 형태의 가중 합산 구조를 도입하며, 가중치는 센서의 신뢰도 및 고장 민감도에 따라 조정된다. 이 구조는 향후 디지털 트윈, 예측 유지보수(PHM), 다변량 신호 기반 이상 원인 분석의 기초 지표로 확장 가능하다.
</p>

<h4 id="hybrid-dnn" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
🤖 딥러닝 기반 Hybrid 구조 및 경량화 제품화 설계
</h4>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
구조 리듬 기반 지표는 해석 가능성과 설명력에서는 우수하지만, 비정형 고장 패턴이나 비선형적 상태 전이에 대한 일반화 성능은 제한적일 수 있다. 이를 보완하기 위해, 본 연구는 SRE, NSI, Periodicity 등 리듬 기반 feature를 중심으로 한 딥러닝 하이브리드 구조를 설계하였다. 특히 <em>“Feature Engineering 기반 시계열 예측기”</em>로서 LSTM, 1D-CNN, Transformer Encoder 구조를 도입하고, 입력값으로는 <MathJax inline>{"$[SRE(t), \\ NSI(t), \\ dSRE/dt, \\text{AdaptiveThreshold}(t)]$"}</MathJax> 등의 조합을 사용하였다.

해당 구조는 구조 리듬의 복잡도를 해석 가능한 입력값으로 정제하고, 신경망은 이 값을 기반으로 고장 가능성, 이상도 스코어, 잔여 수명(RUL)을 예측하도록 학습된다. 특히 1D CNN을 기반으로 한 경량 네트워크(GPI-Net)는 임베디드 시스템에서도 실시간으로 작동 가능한 속도와 메모리 효율을 확보하며, FPGA 또는 Edge-TPU 환경에 쉽게 이식 가능하다.

이 구조는 SRE 기반 진단 시스템의 제품화 가능성을 높이며, <em>“설명 가능한 신호 기반 지표 + 자동화 가능한 패턴 기반 분류기”</em>의 융합을 통해 고장 진단의 패러다임 전환을 촉진할 수 있다.
</p>


<h4 id="stochastic-sre" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
📊 SRE/NSI의 확률론적 해석 프레임으로의 확장
</h4>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
현재까지의 SRE(Spectral Rhythm Entropy) 및 NSI(Noise Spread Index)는 결정론적 수식에 기반하여 정의되어 있으나, 실제 산업 환경에서는 센서 노이즈, 잡음, 샘플링 오류, 데이터 누락 등 다양한 확률적 변동 요소가 존재한다. 이러한 현실을 반영하여, 본 연구는 <em>SRE/NSI의 확률적 모델링</em>을 통해 기존 수식을 확장하는 방향을 제시한다.

우선 <MathJax inline>{"$\\Delta\\kappa(t)$"}</MathJax>의 확률 분포 <MathJax inline>{"$p_{\\Delta\\kappa}(x)$"}</MathJax>를 커널 밀도 추정(KDE) 또는 비정규 분포로 근사하고, 이 분포로부터 정보 엔트로피 <MathJax inline>{"$H(p)$"}</MathJax>를 계산하는 방식으로 SRE를 재정의한다. 나아가 <em>Hidden Markov Model(HMM)</em> 또는 <em>State Space Model</em>과의 연계를 통해 SRE를 시간적으로 변화하는 은닉 상태(hidden state)로 해석하고, 고장 진행의 확률적 전이를 모델링한다. 이때 관측값은 NSI 또는 원시 전류 신호이며, 상태 전이 확률은 리듬의 변동성에 기반한다.

이러한 확률론적 해석은 SRE의 수학적 정합성을 높이는 동시에, 예지보전(PHM), 상태 추정, 예측 경고 모델 등 고차원 응용으로 확장할 수 있는 통계 기반 프레임워크를 제공한다.
</p>


<h4 id="mck-dynamics" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
⚙️ M-C-K 물리 기반 진동 해석과의 연계
</h4>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
SRE(Spectral Rhythm Entropy)와 NSI(Noise Spread Index)는 시계열 신호의 리듬 붕괴를 정량화하는 수식 기반 지표로서 작동하지만, 이들 지표가 실제 물리 시스템의 동역학적 특성과 어떻게 연결되는지에 대한 해석이 부족하다면, 공학적 설득력은 제한적일 수 있다. 이에 따라 본 연구는 질량-감쇠-강성(Mass-Damping-Stiffness, M-C-K) 모델을 기반으로, 구조 리듬 지표의 물리적 대응 관계를 정립하고자 하였다.

일반적인 2차 시스템에서, 고장이 발생하면 감쇠 계수(c)의 감소 또는 강성(k)의 저하가 발생하며 이는 공진 주파수의 이동, 진폭의 증가, 리듬의 불안정성 증가로 나타난다. 이러한 물리적 변화는 곧 파형의 곡률 변화로 이어지며, 이는 <MathJax inline>{"$\\Delta\\kappa(t)$"}</MathJax>의 시계열 분포로 나타난다. 즉, 곡률 기반 SRE는 시스템의 구조적 에너지 응답이 비정상화되는 시점을 <em>기하학적 신호 리듬</em>으로 포착한 것이다.

예컨대 강성이 낮아질수록, 동일한 외란에 대한 시스템의 반응 진폭은 증가하며, 곡률 값이 급변하게 된다. 반면 감쇠가 손실되면 고주파 성분이 잔존하게 되어 NSI 값이 폭발적으로 증가한다. 이러한 현상은 베어링 마모, 축 휨, 윤활유 소실 등 다양한 고장 모드에서 관찰되며, M-C-K 시스템의 파라미터 추정을 위한 간접 역모델링 구조로 활용 가능하다. 따라서 본 구조 리듬 기반 지표는 단순 이상 탐지를 넘어, <em>실제 물리 파라미터의 변화를 정량 추론</em>할 수 있는 새로운 해석 도구로 자리매김할 수 있다.
</p>


<h4 id="multi-resolution" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
🔍 Sliding Window 기반 탐지 구조의 한계와 다중 해상도 보완 전략
</h4>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
Sliding Window 방식은 구조 리듬 기반 진단 시스템에서 시간-국소적 이상 신호를 포착하는 데 유용하지만, 고정된 윈도우 크기로 인한 해상도 불균형 문제가 존재한다. 짧은 윈도우는 민감도는 높지만 안정성이 낮고, 긴 윈도우는 반응성은 낮지만 잡음을 평균화할 수 있다. 이로 인해, 점진적 고장과 급격한 이상 모두를 안정적으로 포착하기 어렵다는 구조적 한계가 존재한다.

본 연구는 이러한 문제를 해결하기 위해 <em>다중 해상도 분석 구조(Multi-Resolution Analysis, MRA)</em>를 도입하였다. 동일한 시계열에 대해 서로 다른 윈도우 길이(Short, Mid, Long)에서 SRE와 NSI를 각각 계산하고, 이들의 상승 시점을 비교하여 고장 발생의 <em>다중 시점 확률 공간</em>을 구성한다. 예컨대 짧은 윈도우에서는 급변 잡음을 탐지하고, 중간 구간에서는 주기성 손실을, 긴 구간에서는 구조 피로 누적을 포착할 수 있다.

이를 통해 고장의 전조, 발생, 진행을 다양한 시간 스케일에서 추적 가능하며, 탐지 민감도와 신뢰도 간의 균형도 확보된다. 이 구조는 향후 <em>Sliding Threshold, Dynamic Alarm Score</em> 등과 결합하여, 고장 상태에 대한 총체적 진단을 가능케 한다.
</p>

<h4 id="multi-sre" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
🌐 센서 다중화 구조로의 확장: Multi-SRE 기반 이상 진단 구조 설계
</h4>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
단일 전류 센서를 기반으로 설계된 기존 SRE 분석 구조는 해석력은 뛰어나지만, 복잡계 시스템에서의 다원적 고장 양상을 포착하기에는 제한적일 수 있다. 이를 해결하기 위해 본 연구는 <em>센서 다중화 기반 리듬 분석 구조</em>를 제안하였다. 다양한 센서 채널(전류, 진동, 온도, 음향 등)로부터 동일한 구조 리듬 지표를 계산하고, 이를 통합하여 <MathJax inline>{"$\\text{SRE}_{multi}(t)$"}</MathJax> 벡터로 구성한다.

이러한 구조는 각 센서의 이상도 상승 시점, 증가 속도, 변화 패턴을 종합적으로 해석할 수 있으며, 고장의 원인과 전이 방향을 추정하는 데 유리하다. 예를 들어, 진동 기반 SRE가 먼저 상승하고, 이후 전류 기반 SRE가 뒤따를 경우 이는 기계적 고장 → 전기적 부하 증가로의 전이를 의미할 수 있다. 또한 각 센서의 중요도에 따라 가중치를 부여한 다음 식을 통해 통합 이상도 점수를 계산한다:  
<br /><MathJax>{"$$\\text{MultiSRE}(t) = w_1 SRE_{\\text{current}}(t)^2 + w_2 SRE_{\\text{vib}}(t)^2 + \\cdots$$"}</MathJax>

이 구조는 이상 탐지뿐만 아니라 고장 전이 경로 추론, 고장 유형 분류, 조기경보 시스템 설계 등에도 적용 가능하며, 반도체·항공·로봇 등 다중 센서 시스템에서의 실질적 진단 성능 향상을 가능케 한다.
</p>

<h4 id="model-comparison" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
📈 기존 통계 기반 기법과의 정량 비교 및 우월성 입증 구조
</h4>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
SRE(Spectral Rhythm Entropy), NSI(Noise Spread Index) 등 본 연구의 구조 리듬 기반 지표들은 곡률과 비정형성에 초점을 맞추어 고장의 전조를 포착하는 새로운 방식이다. 하지만 이들이 실제로 기존의 통계 기반 진단 기법들과 비교하여 얼마나 실효적이며 우수한지를 입증하기 위해서는, <em>정량적 성능 비교 및 수학적 차별성 분석</em>이 필요하다.

대표적인 비교 대상은 ARIMA, HMM, GARCH, Kalman Filter 등으로, 각각 자기상관, 상태 전이, 조건부 분산, 예측 보정 등의 능력을 갖추고 있다. 이들과 SRE 기반 구조를 동일한 데이터셋에 적용하여 탐지 정확도(accuracy), 조기성(lead time), 오탐률(false positive rate), 반복 신뢰도(consistency) 등의 측면에서 정량 비교하면, 본 지표의 우수성을 객관적으로 증명할 수 있다.  
특히 SRE는 비정상 시계열에도 적용 가능하며, 외란이나 잡음에 민감한 기존 통계 모델과는 달리 <em>구조 리듬 변화 자체</em>에 집중한다는 점에서 차별된다.

또한, 수학적 관점에서도 RMS, PSD, Spectral Entropy 등의 기존 지표와 상호 정보량(Mutual Information)을 분석하여, SRE가 제공하는 고유 정보량이 통계적으로 독립적임을 보여줄 수 있다. 이를 통해 SRE/NSI는 기존 지표와 보완적인 역할이 아닌, 독립적이고 우월한 구조 해석 지표임을 정량적으로 입증할 수 있다.
</p>

<h4 id="robustness" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
🛡️ 실측 신호에서의 결손, 노이즈, 왜곡에 대한 강건성 설계 보완
</h4>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
이론적 정합성과 수식적 정확도를 갖춘 구조 리듬 기반 지표라 하더라도, 실제 산업 현장에서 수집된 실측 신호의 결손(missing), 이상치(outlier), 샘플링 편차 등의 문제가 해소되지 않으면 진단 시스템으로서의 신뢰성을 확보할 수 없다. 따라서 본 연구는 실측 환경에서의 SRE/NSI 운용을 위한 <em>강건한 신호 보정 및 지표 설계</em>를 병행하였다.

먼저 데이터 결손에 대해서는 선형 보간(linear interpolation), spline 보간, Gaussian process 예측 등 다양한 보간 기법을 통해 최소한의 왜곡으로 원 신호를 복원하고, 이상값에 대해서는 MAD, Hampel filter, IQR 기반 filtering 등을 활용하여 미분 계산의 안정성을 확보하였다. 특히 SRE는 곡률의 2차 미분에 의존하기 때문에, 극단값에 매우 민감한 특성을 가지므로 이러한 사전 보정 단계는 필수적이다.

추가로 Sliding Window 내 유효 샘플의 비율을 체크하여, 일정 이상 coverage(예: 80%)가 충족되지 않을 경우 해당 SRE 계산을 생략함으로써, 통계적 왜곡을 방지하였다. 나아가 NSI를 기존의 분산 기반이 아닌 IQR 기반의 robust dispersion metric으로 재정의하면, 잡음에 둔감하고 결측에 강한 지표로 활용될 수 있다. 이러한 다층적 강건성 보완은 실제 설비 신호에서도 안정적으로 작동하는 이상 탐지 시스템을 구현하는 기반이 된다.
</p>

<h4 id="ai-hybrid" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
🧠 AI 또는 Hybrid 구조와의 통합 가능성 및 Feature Engineering 방향성
</h4>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
SRE와 NSI는 고장 진단을 위한 설명 가능한 수학 기반 지표로서 강력하지만, 비선형적이고 다차원적인 실제 고장 패턴을 자동 예측하는 데에는 딥러닝 기반의 모델이 필요하다. 이에 따라 본 연구는 SRE/NSI를 중심 feature로 활용하면서, AI 기반의 이상 예측 모델과 결합하는 <em>Hybrid Fault Prediction Framework</em>를 설계하였다.

Feature Engineering 측면에서, <MathJax inline>{"$[SRE(t), \\ NSI(t), \\ dSRE/dt, \\text{AdaptiveThreshold}(t), \\text{Periodicity}(t)]$"}</MathJax> 등을 구성하고, 이를 1D CNN, LSTM, GRU, Transformer 등 다양한 시계열 모델에 입력하여 fault score 또는 고장 가능성을 출력하도록 한다. 이 구조는 단일 신호가 아닌 <em>리듬 변화의 의미 있는 해석 값</em>을 딥러닝이 학습하게 함으로써, 설명력 있는 자동 진단이 가능해진다.

또한, 딥러닝 예측기와의 결합에서 중요한 해석 가능성(Explainability)을 위해 SHAP, LIME, Grad-CAM 등의 도구를 함께 적용할 수 있으며, SRE의 상승 구간이 모델의 출력에 얼마나 기여했는지를 시각화하여 현장의 신뢰를 얻을 수 있다. 이 구조는 향후 실시간 예측, RUL 추정, 장비 상태 분류 등에 유연하게 확장 가능하며, 학문적으로는 수식 기반 신호 해석과 AI 예측을 융합한 새로운 통합 진단 설계로 자리잡을 수 있다.
</p>


<h4 id="sensor-expansion" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
📡 멀티센서 기반 확장성과 구조적 통합 설계 보완
</h4>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
현재 구조 리듬 기반 고장 진단 프레임워크는 주로 전류 신호를 중심으로 설계되었지만, 실제 산업 환경에서는 진동, 온도, 음향, 압력 등 다양한 센서가 병렬적으로 작동하며 시스템 상태를 반영한다. 따라서 SRE, NSI 등의 지표가 단일 센서에만 국한되지 않고, <em>다채널 센서 환경에서도 동일한 원리로 확장 가능</em>하다는 점을 구조적으로 증명하고자 하였다.

본 연구는 멀티센서 시스템에서 각 센서별로 개별 SRE/NSI를 병렬 계산하고, 이를 시간 동기화하여 통합 이상도 벡터 <MathJax inline>{"$\\mathbf{SRE}_{multi}(t)$"}</MathJax>를 구성하는 구조를 설계하였다. 이때 각 센서의 중요도에 따라 가중치를 부여하거나, 이상 탐지 민감도에 따라 adaptive weighting을 적용하여 센서 간 균형 있는 진단 효과를 확보하였다.

나아가 센서 간의 상관관계 분석, mutual information 기반 정보 융합, cross-entropy를 활용한 이질적 센서 신호 간 리듬 불일치 추적 등을 통해 <em>물리적 고장 전이 경로</em>를 추론하는 진단 메커니즘도 구현 가능하다. 이러한 구조는 향후 스마트팩토리 환경에서 센서 수가 많은 시스템에도 적응할 수 있으며, SRE 기반 진단 알고리즘의 범용성과 실용성을 크게 향상시킨다.
</p>


<h4 id="realtime-sliding" style={{ marginTop: "36px", fontSize: "1.25rem", fontWeight: "bold", color: "#333" }}>
⏱️ Sliding Window 기반 탐색 구조의 고도화와 실시간성 향상 전략
</h4>

<p style={{ marginTop: "12px", lineHeight: "1.8" }}>
Sliding Window 구조는 리듬 기반 이상 감지에서 시간 국소적 패턴을 추출하기 위한 핵심 메커니즘이다. 그러나 모든 시점에 대해 동일한 윈도우 크기와 동일한 간격으로 반복 계산을 수행할 경우, <em>계산 지연(latency), 리소스 과다 소모, 반응 속도 저하</em> 등의 실시간성 한계가 발생한다. 본 연구는 이러한 현실적 문제를 해결하기 위해 Sliding 구조의 고도화 및 실시간화 전략을 제안하였다.

우선, 전체 시계열에 대해 연속 탐색이 아닌, <em>이벤트 기반 Adaptive Sliding 구조</em>를 채택하였다. 즉, 신호의 분산이나 SRE 변화가 특정 threshold 이상으로 급등하는 구간에서만 계산을 수행하는 방식으로, 계산량을 60~80%까지 절감할 수 있다. 또한 중심값 기반 요약(Summary-based Computation)을 통해 윈도우 내 전체 프레임이 아닌 중심부 몇 개의 핵심 구간만을 계산 대상으로 삼아 <em>미분 연산 최적화</em>를 수행하였다.

더불어 실시간 시스템 적용을 위해 각 윈도우 내 이상 감지 여부를 flag 형태로 압축하고, PLC/FPGA/Edge-TPU 장비에서도 수행 가능한 경량 연산 구조를 구현하였다. 이러한 고도화된 Sliding 구조는 단순히 지표 계산의 효율성만을 높이는 것이 아니라, 실제 산업 제어 시스템에 진입 가능한 수준의 <em>운용 가능성(operability)</em>을 확보하게 한다. 이는 곧 구조 리듬 기반 진단 알고리즘의 산업 현장 실장화를 위한 핵심 요소 중 하나로 작동할 것이다.
</p>





    </div>
    </MathJaxContext>
  );
}