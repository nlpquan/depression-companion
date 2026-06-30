'use client';

import { useState } from 'react';
import MoodChart from '@/components/MoodChart';
import JournalEditor from '@/components/JournalEditor';
import AudioRecorder from '@/components/AudioRecorder';
import RiskIndicator from '@/components/RiskIndicator';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<'journal' | 'voice' | 'forecast'>('journal');
  const [analysis, setAnalysis] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const analyzeText = async (text: string) => {
    setLoading(true);
    try {
      const response = await fetch('${API_URL}/api/v1/analyze/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await response.json();
      setAnalysis(data);
    } catch (error) {
      console.error('Analysis failed:', error);
    }
    setLoading(false);
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Tabs */}
      <div className="flex gap-2 mb-8 bg-white rounded-lg p-1 shadow-sm">
        {[
          { key: 'journal', label: '📝 Journal' },
          { key: 'voice', label: '🎤 Voice' },
          { key: 'forecast', label: '📊 Forecast' },
        ].map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key as any)}
            className={`flex-1 py-3 px-4 rounded-md text-sm font-medium transition ${
              activeTab === tab.key
                ? 'bg-blue-600 text-white shadow'
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="grid md:grid-cols-3 gap-6">
        {/* Main Panel */}
        <div className="md:col-span-2 space-y-6">
          {activeTab === 'journal' && (
            <JournalEditor onAnalyze={analyzeText} loading={loading} />
          )}
          {activeTab === 'voice' && <AudioRecorder />}
          {activeTab === 'forecast' && <MoodChart />}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {analysis && (
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h3 className="font-semibold text-lg mb-4">Analysis Results</h3>
              
              <div className="space-y-4">
                <ScoreBar
                  label="Depression"
                  score={analysis.scores.depression_score}
                  color="blue"
                />
                <ScoreBar
                  label="Anxiety"
                  score={analysis.scores.anxiety_score}
                  color="yellow"
                />
                <ScoreBar
                  label="Mood"
                  score={analysis.scores.mood_score}
                  color="green"
                  invert
                />
              </div>
              
              <div className="mt-4 pt-4 border-t text-sm text-gray-500">
                Confidence: {(analysis.scores.confidence * 100).toFixed(0)}%
                <br />
                Time: {analysis.processing_time_ms.toFixed(0)}ms
              </div>
            </div>
          )}

          <RiskIndicator />
          
          <div className="bg-white rounded-xl shadow-sm p-6">
            <h3 className="font-semibold mb-3">Quick Tips</h3>
            <ul className="text-sm text-gray-600 space-y-2">
              <li>• Try journaling daily for best insights</li>
              <li>• Voice analysis works best in quiet spaces</li>
              <li>• Check your 7-day forecast regularly</li>
              <li>• Reach out if risk level increases</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

function ScoreBar({
  label,
  score,
  color,
  invert = false,
}: {
  label: string;
  score: number;
  color: string;
  invert?: boolean;
}) {
  const displayScore = invert ? 1 - score : score;
  const percentage = (displayScore * 100).toFixed(0);
  
  const colorMap: Record<string, string> = {
    blue: 'bg-blue-500',
    yellow: 'bg-yellow-500',
    green: 'bg-green-500',
    red: 'bg-red-500',
  };

  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span>{label}</span>
        <span className="font-medium">{percentage}%</span>
      </div>
      <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${colorMap[color]}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}