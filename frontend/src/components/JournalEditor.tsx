'use client';

import { useState } from 'react';

interface Props {
  onAnalyze: (text: string) => void;
  loading: boolean;
}

export default function JournalEditor({ onAnalyze, loading }: Props) {
  const [text, setText] = useState('');
  const [wordCount, setWordCount] = useState(0);

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    setText(value);
    setWordCount(value.trim() ? value.trim().split(/\s+/).length : 0);
  };

  const handleAnalyze = () => {
    if (text.trim()) {
      onAnalyze(text);
    }
  };

  const prompts = [
    "How are you feeling today?",
    "What's been on your mind lately?",
    "Describe your energy levels today.",
    "What are you grateful for?",
  ];

  return (
    <div className="bg-white rounded-xl shadow-sm p-6">
      <h3 className="text-lg font-semibold mb-4">Daily Journal</h3>

      {/* Prompts */}
      <div className="flex flex-wrap gap-2 mb-4">
        {prompts.map((prompt) => (
          <button
            key={prompt}
            onClick={() => setText(prompt + ' ')}
            className="text-xs px-3 py-1.5 bg-gray-100 hover:bg-gray-200 rounded-full text-gray-600 transition"
          >
            {prompt}
          </button>
        ))}
      </div>

      {/* Text area */}
      <textarea
        value={text}
        onChange={handleTextChange}
        placeholder="Start writing about your day..."
        className="w-full h-40 p-4 border rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
        maxLength={5000}
      />

      {/* Footer */}
      <div className="flex justify-between items-center mt-3">
        <span className="text-sm text-gray-400">
          {wordCount} words • {text.length}/5000
        </span>
        <button
          onClick={handleAnalyze}
          disabled={!text.trim() || loading}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <span className="animate-spin">⏳</span> Analyzing...
            </span>
          ) : (
            'Analyze'
          )}
        </button>
      </div>
    </div>
  );
}