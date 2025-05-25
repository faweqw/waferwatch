// src/utils/loadChartData.js
export const loadChartData = (json, key = 'raw') => {
    if (!json || !json[key] || !json[key].datasets || !json[key].datasets[0]) {
      console.warn(`No data found for key: ${key}`);
      return null;
    }

    const entry = json[key];
    const labels = entry.labels || [];
    const dataset = entry.datasets[0];
    const data = Array.isArray(dataset.data) ? dataset.data : [];

    return {
      labels,
      datasets: [{
        label: key.toUpperCase(),
        data,
        borderColor: dataset.borderColor || 'rgba(75,192,192,1)',
        fill: dataset.fill ?? false,
        tension: dataset.tension ?? 0.3
      }]
    };
  };
