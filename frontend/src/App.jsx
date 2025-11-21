import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';
import Detector from './components/Detector';
import History from './components/History';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  return (
    <div className="min-h-screen bg-cyber-black text-white selection:bg-neon-blue selection:text-black">
      <div className="fixed inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 pointer-events-none z-0"></div>
      <div className="relative z-10">
        <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

        <main className="container mx-auto px-4 py-8">
          {activeTab === 'dashboard' && <Dashboard setActiveTab={setActiveTab} />}
          {activeTab === 'detector' && <Detector />}
          {activeTab === 'history' && <History />}
        </main>
      </div>
    </div>
  );
}

export default App;
