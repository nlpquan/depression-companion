'use client';

export default function RiskIndicator() {
  // This would fetch real data in production
  const riskLevel = 'low';

  const config = {
    low: {
      color: 'bg-green-50 border-green-200',
      text: 'text-green-800',
      icon: '✅',
      label: 'Low Risk',
      description: 'No warning signals detected. Keep up the good habits!',
    },
    medium: {
      color: 'bg-yellow-50 border-yellow-200',
      text: 'text-yellow-800',
      icon: '⚠️',
      label: 'Medium Risk',
      description: 'Some warning signals detected. Consider reaching out.',
    },
    high: {
      color: 'bg-red-50 border-red-200',
      text: 'text-red-800',
      icon: '🚨',
      label: 'High Risk',
      description: 'Multiple warning signals. Please seek support.',
    },
  };

  const current = config[riskLevel];

  return (
    <div className={`rounded-xl border p-6 ${current.color}`}>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-2xl">{current.icon}</span>
        <h3 className={`font-semibold ${current.text}`}>{current.label}</h3>
      </div>
      <p className="text-sm text-gray-600">{current.description}</p>
    </div>
  );
}