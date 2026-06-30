"use client";

import { useRouter } from 'next/navigation';

export default function Home() {
  const router = useRouter();

  return (
    <main className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      <div className="max-w-6xl mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            Depression Companion
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Multimodal AI-powered depression detection and monitoring system.
            Analyze speech patterns and text to gain clinical-grade insights.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <FeatureCard title="Voice Analysis" description="Record or upload speech for real-time prosody and energy analysis" icon="🎤" />
          <FeatureCard title="Text Journal" description="Write about your day and get sentiment and linguistic analysis" icon="📝" />
          <FeatureCard title="Mood Tracking" description="7-day mood forecasts with early warning signals" icon="📊" />
        </div>

        <div className="bg-white rounded-2xl shadow-lg p-8 text-center">
          <h2 className="text-2xl font-semibold mb-6">Get Started</h2>
          <div className="grid md:grid-cols-2 gap-4">
            <button 
              onClick={() => router.push('/dashboard')}
              className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition"
            >
              Start Voice Check-in
            </button>
            <button 
              onClick={() => router.push('/dashboard')}
              className="border-2 border-blue-600 text-blue-600 px-6 py-3 rounded-lg hover:bg-blue-50 transition"
            >
              Write Journal Entry
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}

function FeatureCard({ title, description, icon }: {
  title: string;
  description: string;
  icon: string;
}) {
  return (
    <div className="bg-white rounded-xl shadow-md p-6 hover:shadow-lg transition">
      <div className="text-4xl mb-4">{icon}</div>
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <p className="text-gray-600">{description}</p>
    </div>
  );
}