import React from 'react';
import ChartTabs from './components/ChartTabs';

function App() {
  return (
    <div style={{ padding: 24 }}>
      <h1 style={{ fontSize: 24, fontWeight: 'bold' }}>Waferwatch 대시보드</h1>
      <ChartTabs />
    </div>
  );
}

export default App;
