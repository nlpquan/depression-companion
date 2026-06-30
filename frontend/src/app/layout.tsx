import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Depression Companion',
  description: 'AI-powered depression detection and monitoring system',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-50">
        <nav className="bg-white shadow-sm border-b">
          <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
            <h1 className="text-xl font-bold text-gray-900">
              🧠 Depression Companion
            </h1>
            <div className="flex gap-4 text-sm">
              <a href="/" className="text-gray-600 hover:text-gray-900">Home</a>
              <a href="/dashboard" className="text-gray-600 hover:text-gray-900">Dashboard</a>
            </div>
          </div>
        </nav>
        {children}
      </body>
    </html>
  );
}