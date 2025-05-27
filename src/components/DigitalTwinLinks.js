// src/components/DigitalTwinLinks.js
import React from 'react';

export default function DigitalTwinLinks() {
    const companies = [
        {
          name: 'Siemens',
          description: `산업 자동화 분야에서 디지털 트윈의 절대 강자입니다. "Siemens Xcelerator" 플랫폼을 통해 기계 설계부터 운영까지 물리 기반 시뮬레이션을 정밀하게 구현하며, BMW 공장 등의 실제 제조 환경에 적용되어 생산성을 크게 향상시켰습니다.`,
          image: '/images/31.webp'
        },
        {
          name: 'AWS',
          description: `클라우드 인프라 기반의 디지털 트윈 구축을 선도하고 있습니다. "AWS IoT TwinMaker"를 통해 실시간 센서 기반 트윈을 구성하고, 확장성과 머신러닝 융합이 용이하여 제조 및 플랜트 진단에서 빠르게 확산되고 있습니다.`,
          image: '/images/32.png'
        },
        {
          name: 'Microsoft',
          description: `Azure Digital Twins 플랫폼을 기반으로 스마트 빌딩, 공장, 도시 인프라 등에 활용됩니다. 객체 관계 기반의 구조적 트윈 모델링을 제공하며, Heathrow 공항 운항 최적화 등 엔터프라이즈 수준의 적용 사례가 많습니다.`,
          image: '/images/33.png'
        },
        {
          name: 'Tesla',
          description: `전기차와 자율주행 차량 각각에 대한 디지털 트윈을 개별적으로 운영하여 고장 진단, 주행 데이터 분석, 배터리 상태 예측 등을 실시간으로 수행합니다. 제품 중심의 트윈 기술을 최초로 상용화한 기업입니다.`,
          image: '/images/35.jpg'
        },
        {
          name: 'GE Digital',
          description: `산업용 IoT 분야의 리더로, "Predix" 플랫폼을 활용해 항공기 엔진, 터빈, 공장 설비의 디지털 트윈을 구현하고 있으며, 예측 유지보수 및 수명 주기 분석에 강점을 가지고 있습니다.`,
          image: '/images/36.png'
        },
        {
          name: 'PTC',
          description: `CAD 및 PLM의 강자인 PTC는 "ThingWorx" 플랫폼을 기반으로 IoT 연계 디지털 트윈을 구축하며, 제조 현장의 작업자 지원 및 설비 시뮬레이션을 실현하고 있습니다.`,
          image: '/images/38.png'
        }
      ];
      
      const articles = [
        {
          title: '삼성SDS – 디지털 트윈 기반 혁신 전략',
          url: 'https://www.samsungsds.com/kr/insights/digital_twin_for_innovations.html',
          image: '/images/21.jpg'
        },
        {
          title: 'SK – 스마트 제조를 위한 DT 기술 적용 사례 ①',
          url: 'https://www.skcc.co.kr/insight/trend/3000',
          image: '/images/22.png'
        },
        {
          title: 'SK – 디지털 전환 핵심 전략으로서 DT ②',
          url: 'https://www.skcc.co.kr/insight/trend/2620',
          image: '/images/56.png'
        },
        {
          title: '현대차 – 디지털 트윈으로 미래차 개발 혁신',
          url: 'https://www.hyundai.co.kr/story/CONT0000000000122330',
          image: '/images/24.jpg'
        },
        {
          title: 'IBM – 디지털 트윈과 AI 기반 제조 혁신',
          url: 'https://www.hankyung.com/article/2021020883891',
          image: '/images/45.jpg'
        },
        {
          title: 'NVIDIA – Omniverse로 구축하는 실시간 디지털 트윈',
          url: 'https://www.nvidia.com/ko-kr/omniverse/digital-twins/siemens/',
          image: '/images/41.jpg'
        }
      ];
      
  return (
    <>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px', marginTop: '40px' }}>
        {companies.map((c, index) => (
          <div key={index} style={{
            border: '1px solid #ccc',
            borderRadius: '10px',
            padding: '16px',
            boxShadow: '0 2px 6px rgba(0,0,0,0.1)',
            background: '#fff'
          }}>
            <img
              src={c.image}
              alt={c.name}
              style={{ width: '100%', height: 'auto', maxHeight: '160px', objectFit: 'contain', marginBottom: '12px' }}
            />
            <h4 style={{ fontSize: '1.1rem', marginBottom: '8px' }}>{c.name}</h4>
            <p style={{ fontSize: '0.95rem', color: '#444', lineHeight: '1.6' }}>{c.description}</p>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px', marginTop: '48px' }}>
        {articles.map((article, index) => (
          <div key={index} style={{
            border: '1px solid #ccc',
            borderRadius: '10px',
            padding: '16px',
            boxShadow: '0 2px 6px rgba(0,0,0,0.1)',
            background: '#fff'
          }}>
            <img
              src={article.image}
              alt={article.title}
              style={{ width: '100%', height: 'auto', maxHeight: '180px', objectFit: 'contain', marginBottom: '12px' }}
            />
            <a href={article.url} target="_blank" rel="noopener noreferrer" style={{
              fontSize: '1rem',
              fontWeight: 'bold',
              textDecoration: 'none',
              color: '#004080'
            }}>
              🔗 {article.title}
            </a>
          </div>
        ))}
      </div>
    </>
  );

  
}
