import React, { useEffect, useState } from 'react';
import { ShieldCheck, AlertTriangle, FileText, Video, Mic, Image as ImageIcon, ArrowRight, Activity } from 'lucide-react';
import axios from 'axios';

const getApiUrl = () => {
    const url = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    return url.startsWith('http') ? url : `https://${url}`;
};
const API_URL = getApiUrl();

const StatCard = ({ title, value, icon: Icon, color }) => (
    <div className="bg-cyber-gray border border-gray-800 p-6 rounded-xl hover:border-gray-600 transition-all duration-300 group">
        <div className="flex justify-between items-start mb-4">
            <div className={`p-3 rounded-lg bg-${color}/10 text-${color} group-hover:shadow-[0_0_15px_rgba(0,0,0,0.3)] transition-shadow`}>
                <Icon className="w-6 h-6" />
            </div>
            <span className="text-xs font-mono text-gray-500">LIFETIME</span>
        </div>
        <h3 className="text-3xl font-bold font-mono mb-1">{value}</h3>
        <p className="text-gray-400 text-sm">{title}</p>
    </div>
);

const Dashboard = ({ setActiveTab }) => {
    const [stats, setStats] = useState({
        total_scans: 0,
        threats_detected: 0,
        authentic_media: 0,
        avg_confidence: 0
    });
    const [recentActivity, setRecentActivity] = useState([]);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const statsRes = await axios.get(`${API_URL}/stats`);
                setStats(statsRes.data);

                const historyRes = await axios.get(`${API_URL}/history`);
                setRecentActivity(historyRes.data.slice(0, 5)); // Get top 5
            } catch (error) {
                console.error("Failed to fetch dashboard data:", error);
            }
        };

        fetchData();
        // Poll every 5 seconds for updates
        const interval = setInterval(fetchData, 5000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="space-y-8 animate-fade-in">
            <div className="flex justify-between items-end">
                <div>
                    <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-white to-gray-500 bg-clip-text text-transparent">
                        System Overview
                    </h1>
                    <p className="text-gray-400">Deepfake Detection Monitoring</p>
                </div>
                <button
                    onClick={() => setActiveTab('detector')}
                    className="bg-cyan-400 text-black px-6 py-3 rounded-lg font-bold hover:bg-cyan-300 transition-colors flex items-center space-x-2 shadow-neon-blue opacity-100"
                >
                    <span>START NEW SCAN</span>
                    <ArrowRight className="w-4 h-4" />
                </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <StatCard title="Total Scans" value={stats.total_scans} icon={Activity} color="neon-blue" />
                <StatCard title="Threats Detected" value={stats.threats_detected} icon={AlertTriangle} color="neon-red" />
                <StatCard title="Authentic Media" value={stats.authentic_media} icon={ShieldCheck} color="neon-green" />
                <StatCard title="Avg. Confidence" value={`${stats.avg_confidence}%`} icon={Activity} color="purple-500" />
            </div>



            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="bg-cyber-gray border border-gray-800 rounded-xl p-6">
                    <h3 className="text-xl font-bold mb-6 flex items-center space-x-2">
                        <Activity className="w-5 h-5 text-neon-blue" />
                        <span>Recent Activity</span>
                    </h3>
                    <div className="space-y-4">
                        {recentActivity.length === 0 ? (
                            <p className="text-gray-500 text-center py-4">No recent activity found.</p>
                        ) : (
                            recentActivity.map((item) => (
                                <div key={item.id} className="flex items-center justify-between p-4 bg-black/20 rounded-lg border border-gray-800/50">
                                    <div className="flex items-center space-x-4">
                                        <div className={`w-2 h-2 rounded-full ${item.result.is_fake ? 'bg-neon-red' : 'bg-neon-green'}`}></div>
                                        <div>
                                            <p className="font-medium capitalize">{item.type} Analysis</p>
                                            <p className="text-sm text-gray-500">{new Date(item.timestamp).toLocaleTimeString()}</p>
                                        </div>
                                    </div>
                                    <span className={`px-3 py-1 rounded-full text-xs font-mono border ${item.result.is_fake
                                        ? 'bg-neon-red/10 text-neon-red border-neon-red/20'
                                        : 'bg-neon-green/10 text-neon-green border-neon-green/20'
                                        }`}>
                                        {item.result.is_fake ? 'FAKE' : 'AUTHENTIC'}
                                    </span>
                                </div>
                            ))
                        )}
                    </div>
                </div>

                <div className="bg-gradient-to-br from-indigo-900/20 to-purple-900/20 border border-indigo-500/30 rounded-xl p-6 relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-64 h-64 bg-neon-blue/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2"></div>
                    <h3 className="text-xl font-bold mb-4">Project Highlights</h3>

                    <div className="grid grid-cols-2 gap-4 mb-6">
                        <div className="bg-black/40 p-3 rounded-lg border border-white/5">
                            <div className="text-2xl font-bold text-white">98%</div>
                            <div className="text-neon-blue text-xs font-mono">ACCURACY</div>
                        </div>
                        <div className="bg-black/40 p-3 rounded-lg border border-white/5">
                            <div className="text-2xl font-bold text-white">140k+</div>
                            <div className="text-neon-green text-xs font-mono">IMAGES</div>
                        </div>
                        <div className="bg-black/40 p-3 rounded-lg border border-white/5">
                            <div className="text-2xl font-bold text-white">4</div>
                            <div className="text-purple-500 text-xs font-mono">MODALITIES</div>
                        </div>
                        <div className="bg-black/40 p-3 rounded-lg border border-white/5">
                            <div className="text-xl font-bold text-white">Real-Time</div>
                            <div className="text-neon-red text-xs font-mono">SPEED</div>
                        </div>
                    </div>

                    <h4 className="text-sm font-bold text-gray-400 mb-3 uppercase tracking-wider">Supported Modalities</h4>
                    <div className="grid grid-cols-2 gap-4">
                        {[
                            { label: 'Image', icon: ImageIcon },
                            { label: 'Video', icon: Video },
                            { label: 'Audio', icon: Mic },
                            { label: 'Text', icon: FileText },
                        ].map((m) => (
                            <div key={m.label} className="bg-black/40 p-4 rounded-lg border border-white/5 hover:border-neon-blue/50 transition-colors cursor-default">
                                <m.icon className="w-6 h-6 text-gray-400 mb-2" />
                                <p className="font-mono text-sm">{m.label}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
