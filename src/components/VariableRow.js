// src/components/VariableRow.js
import React from 'react';

export default function VariableRow({ data }) {
  const {
    label,
    color,
    variableImage,
    deviceImage1,
    deviceImage2,
    description,
    device1Desc,
    device2Desc
  } = data;

  const imageStyle = {
    width: '100%',
    maxWidth: '160px',
    borderRadius: '8px',
    marginBottom: '8px'
  };

  const textStyle = {
    fontSize: '14px',
    whiteSpace: 'pre-wrap',
    fontFamily: 'Hanna11, sans-serif',
    color: '#333'
  };

  return (
    <table
      style={{
        width: '100%',
        borderCollapse: 'separate',
        borderSpacing: '24px 12px',
        marginBottom: '40px',
        background: '#fafafa',
        borderRadius: '12px',
        boxShadow: '4px 4px 12px #ccc, -4px -4px 12px #fff',
        padding: '16px'
      }}
    >
      <tbody>
        <tr>
          <td style={{ textAlign: 'center' }}>
            <img src={variableImage} alt="변수 이미지" style={imageStyle} />
          </td>
          <td style={{ textAlign: 'center' }}>
            <img src={deviceImage1} alt="대표 장비1" style={imageStyle} />
          </td>
          <td style={{ textAlign: 'center' }}>
            <img src={deviceImage2} alt="대표 장비2" style={imageStyle} />
          </td>
        </tr>
        <tr>
          <td style={textStyle}>
            <strong style={{ color }}>{label}</strong><br />
            {description}
          </td>
          <td style={textStyle}>
            📌 <b>장비1 설명:</b><br />
            {device1Desc}
          </td>
          <td style={textStyle}>
            📌 <b>장비2 설명:</b><br />
            {device2Desc}
          </td>
        </tr>
      </tbody>
    </table>
  );
}
