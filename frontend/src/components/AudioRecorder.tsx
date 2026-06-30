'use client';

import { useState, useRef } from 'react';

export default function AudioRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [audioURL, setAudioURL] = useState<string | null>(null);
  const [duration, setDuration] = useState(0);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream);

      mediaRecorder.current.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.current.push(e.data);
        }
      };

      mediaRecorder.current.onstop = () => {
        const blob = new Blob(chunks.current, { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);
        setAudioURL(url);
        chunks.current = [];
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.current.start();
      setIsRecording(true);

      // Timer
      const startTime = Date.now();
      timerRef.current = setInterval(() => {
        setDuration(Math.floor((Date.now() - startTime) / 1000));
      }, 1000);
    } catch (error) {
      alert('Microphone access denied. Please allow microphone access.');
      console.error(error);
    }
  };

  const stopRecording = () => {
    mediaRecorder.current?.stop();
    setIsRecording(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm p-6">
      <h3 className="text-lg font-semibold mb-4">Voice Analysis</h3>

      <div className="text-center py-12">
        {!audioURL ? (
          <>
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={`w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-4 transition-all ${
                isRecording
                  ? 'bg-red-500 animate-pulse'
                  : 'bg-blue-600 hover:bg-blue-700'
              }`}
            >
              <span className="text-3xl">{isRecording ? '⏹' : '🎤'}</span>
            </button>
            <p className="text-gray-500">
              {isRecording
                ? `Recording... ${duration}s (click to stop)`
                : 'Click to start recording (max 30s)'}
            </p>
          </>
        ) : (
          <>
            <audio controls src={audioURL} className="w-full mb-4" />
            <button
              onClick={() => {
                setAudioURL(null);
                setDuration(0);
              }}
              className="text-blue-600 hover:text-blue-700"
            >
              Record Again
            </button>
          </>
        )}
      </div>

      {isRecording && (
        <div className="flex justify-center gap-1 items-end h-12">
          {[...Array(20)].map((_, i) => (
            <div
              key={i}
              className="w-1 bg-blue-500 rounded-full animate-pulse"
              style={{
                height: `${Math.random() * 100}%`,
                animationDelay: `${i * 0.1}s`,
              }}
            />
          ))}f
        </div>
      )}
    </div>
  );
}