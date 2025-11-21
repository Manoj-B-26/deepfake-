import React, { useEffect, useState } from 'react';
import { Clock, File, Search } from 'lucide-react';
import axios from 'axios';

const History = () => {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const res = await axios.get(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/history`);
                setHistory(res.data);
            } catch (err) {
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        fetchHistory();
    }, []);

    return (
        <div className="max-w-4xl mx-auto animate-fade-in">
            <div className="flex justify-between items-center mb-8">
                <h2 className="text-3xl font-bold">Scan History</h2>
                <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                    <input
                        type="text"
                        placeholder="Search logs..."
                        className="bg-cyber-gray border border-gray-800 rounded-lg pl-10 pr-4 py-2 text-sm focus:border-neon-blue focus:outline-none w-64"
                    />
                </div>
            </div>

            <div className="bg-cyber-gray border border-gray-800 rounded-xl overflow-hidden">
                <div className="grid grid-cols-12 gap-4 p-4 border-b border-gray-800 text-sm font-mono text-gray-500">
                    <div className="col-span-4">FILE / SOURCE</div>
                    <div className="col-span-2">TYPE</div>
                    <div className="col-span-3">RESULT</div>
                    <div className="col-span-3 text-right">TIMESTAMP</div>
                </div>

                {loading ? (
                    <div className="p-8 text-center text-gray-500">Loading history...</div>
                ) : history.length === 0 ? (
                    <div className="p-8 text-center text-gray-500">No scan history found.</div>
                ) : (
                    <div className="divide-y divide-gray-800">
                        {history.map((item) => (
                            <div key={item.id} className="grid grid-cols-12 gap-4 p-4 hover:bg-white/5 transition-colors items-center">
                                <div className="col-span-4 flex items-center space-x-3">
                                    <div className="p-2 bg-gray-800 rounded">
                                        <File className="w-4 h-4 text-gray-400" />
                                    </div>
                                    <span className="truncate font-medium">{item.filename}</span>
                                </div>
                                <div className="col-span-2">
                                    <span className="px-2 py-1 rounded text-xs font-mono bg-gray-800 text-gray-300 uppercase">
                                        {item.type}
                                    </span>
                                </div>
                                <div className="col-span-3">
                                    <span className={`px-2 py-1 rounded text-xs font-bold ${item.result.is_fake
                                        ? 'bg-neon-red/10 text-neon-red border border-neon-red/20'
                                        : 'bg-neon-green/10 text-neon-green border border-neon-green/20'
                                        }`}>
                                        {item.result.is_fake ? 'FAKE DETECTED' : 'AUTHENTIC'}
                                    </span>
                                </div>
                                <div className="col-span-3 text-right text-gray-500 text-sm font-mono">
                                    {new Date(item.timestamp).toLocaleString()}
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

export default History;
