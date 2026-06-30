'use client';

import { useEffect, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts';

export default function MoodChart() {
  const [forecast, setForecast] = useState<any>(null);

  useEffect(() => {
    fetch('/api/v1/forecast/demo-user')
      .then((res) => res.json())
      .then(setForecast)
      .catch(console.error);
  }, []);

  if (!forecast) {
    return (
      <div className="bg-white rounded-xl shadow-sm p-6">
        <p className="text-gray-500">Loading forecast...</p>
      </div>
    );
  }

  // Build chart data
  const chartData = [
    {
      day: 'Today',
      mood: 0.65,
      lower: 0.60,
      upper: 0.70,
    },
    {
      day: 'Day 1',
      mood: forecast.trajectory.day_1,
      lower: forecast.lower_bound.day_1,
      upper: forecast.upper_bound.day_1,
    },
    {
      day: 'Day 3',
      mood: forecast.trajectory.day_3,
      lower: forecast.lower_bound.day_3,
      upper: forecast.upper_bound.day_3,
    },
    {
      day: 'Day 7',
      mood: forecast.trajectory.day_7,
      lower: forecast.lower_bound.day_7,
      upper: forecast.upper_bound.day_7,
    },
  ];

  const riskColor =
    forecast.risk_level === 'high'
      ? 'text-red-600'
      : forecast.risk_level === 'medium'
      ? 'text-yellow-600'
      : 'text-green-600';

  return (
    <div className="bg-white rounded-xl shadow-sm p-6">
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-lg font-semibold">7-Day Mood Forecast</h3>
        <span className={`font-bold capitalize ${riskColor}`}>
          {forecast.risk_level} Risk
        </span>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="day" />
          <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
          <Tooltip
            formatter={(value: number) => `${(value * 100).toFixed(0)}%`}
          />
          {/* Confidence interval */}
          <Area
            type="monotone"
            dataKey="upper"
            stroke="none"
            fill="#3b82f6"
            fillOpacity={0.1}
          />
          <Area
            type="monotone"
            dataKey="lower"
            stroke="none"
            fill="#3b82f6"
            fillOpacity={0.1}
          />
          {/* Mood line */}
          <Line
            type="monotone"
            dataKey="mood"
            stroke="#3b82f6"
            strokeWidth={3}
            dot={{ r: 5, fill: '#3b82f6' }}
          />
        </AreaChart>
      </ResponsiveContainer>

      {forecast.active_warnings?.length > 0 && (
        <div className="mt-4 p-4 bg-yellow-50 rounded-lg">
          <p className="font-medium text-yellow-800 mb-2">
            ⚠️ Active Warning Signals
          </p>
          <ul className="text-sm text-yellow-700 space-y-1">
            {forecast.active_warnings.map((w: any, i: number) => (
              <li key={i}>• {w.description}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}