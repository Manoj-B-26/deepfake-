import React from 'react';
import { Shield, Activity, History as HistoryIcon, Menu } from 'lucide-react';

const Navbar = ({ activeTab, setActiveTab }) => {
    const navItems = [
        { id: 'dashboard', label: 'Dashboard', icon: Activity },
        { id: 'detector', label: 'Detector', icon: Shield },
        { id: 'history', label: 'History', icon: HistoryIcon },
    ];

    return (
        <nav className="border-b border-cyber-gray bg-cyber-black/80 backdrop-blur-md sticky top-0 z-50">
            <div className="container mx-auto px-4">
                <div className="flex items-center justify-between h-16">
                    <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('dashboard')}>
                        <div className="w-10 h-10 bg-gradient-to-br from-neon-blue to-blue-600 rounded-lg flex items-center justify-center shadow-neon-blue">
                            <Shield className="w-6 h-6 text-black" />
                        </div>
                        <span className="text-xl font-bold tracking-wider font-mono">
                            DEEP<span className="text-neon-blue">TRUTH</span>_AI
                        </span>
                    </div>

                    <div className="hidden md:flex space-x-8">
                        {navItems.map((item) => (
                            <button
                                key={item.id}
                                onClick={() => setActiveTab(item.id)}
                                className={`flex items-center space-x-2 px-3 py-2 rounded-md transition-all duration-300 ${activeTab === item.id
                                        ? 'text-neon-blue bg-neon-blue/10 shadow-[0_0_15px_rgba(0,243,255,0.2)]'
                                        : 'text-gray-400 hover:text-white hover:bg-white/5'
                                    }`}
                            >
                                <item.icon className="w-4 h-4" />
                                <span className="font-mono text-sm">{item.label}</span>
                            </button>
                        ))}
                    </div>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
