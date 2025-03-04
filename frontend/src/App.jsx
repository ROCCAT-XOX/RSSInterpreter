import React, { useState } from 'react';
import RSSFeedAnalyzer from './components/RSSFeedAnalyzer';
import TrainingMonitor from './components/TrainingMonitor';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('analyzer');

  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex justify-center h-16">
            <div className="flex space-x-8">
              <button
                className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium ${
                  activeTab === 'analyzer' 
                    ? 'border-indigo-500 text-gray-900' 
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
                onClick={() => setActiveTab('analyzer')}
              >
                RSS Analyzer
              </button>
              <button
                className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium ${
                  activeTab === 'training' 
                    ? 'border-indigo-500 text-gray-900' 
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
                onClick={() => setActiveTab('training')}
              >
                Training Monitor
              </button>
            </div>
          </div>
        </div>
      </nav>

      <main className="py-10">
        {activeTab === 'analyzer' ? <RSSFeedAnalyzer /> : <TrainingMonitor />}
      </main>
    </div>
  );
}

export default App;