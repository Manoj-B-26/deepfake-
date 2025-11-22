import React, { useEffect, useState } from 'react';
import { Clock, File, Search } from 'lucide-react';
import axios from 'axios';

const History = () => {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedItem, setSelectedItem] = useState(null);

    const getApiUrl = () => {
        const url = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        return url.startsWith('http') ? url : `https://${url}`;
    };
    const API_URL = getApiUrl();

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const res = await axios.get(`${API_URL}/history`);
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
                            <div
                                key={item.id}
                                onClick={() => setSelectedItem(item)}
                                className="grid grid-cols-12 gap-4 p-4 hover:bg-white/5 transition-colors items-center cursor-pointer"
                            >
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

            {/* Details Modal */}
            {selectedItem && (
                <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4" onClick={() => setSelectedItem(null)}>
                    <div className="bg-cyber-gray border border-gray-700 rounded-xl max-w-2xl w-full p-6 shadow-2xl relative" onClick={e => e.stopPropagation()}>
                        <button
                            onClick={() => setSelectedItem(null)}
                            className="absolute top-4 right-4 text-gray-400 hover:text-white"
                        >
                            ✕
                        </button>

                        <h3 className="text-2xl font-bold mb-1">Scan Details</h3>
                        <p className="text-gray-400 text-sm mb-6 font-mono">{selectedItem.id}</p>

                        <div className="grid grid-cols-2 gap-6 mb-6">
                            <div className="bg-black/30 p-4 rounded-lg border border-gray-800">
                                <div className="text-sm text-gray-500 mb-1">FILE</div>
                                <div className="font-medium truncate" title={selectedItem.filename}>{selectedItem.filename}</div>
                            </div>
                            <div className="bg-black/30 p-4 rounded-lg border border-gray-800">
                                <div className="text-sm text-gray-500 mb-1">TIMESTAMP</div>
                                <div className="font-medium">{new Date(selectedItem.timestamp).toLocaleString()}</div>
                            </div>
                        </div>

                        <div className={`p-4 rounded-lg border mb-6 ${selectedItem.result.is_fake
                            ? 'bg-neon-red/10 border-neon-red/30'
                            : 'bg-neon-green/10 border-neon-green/30'}`}>
                            <div className="flex justify-between items-center mb-2">
                                <span className={`font-bold text-lg ${selectedItem.result.is_fake ? 'text-neon-red' : 'text-neon-green'}`}>
                                    {selectedItem.result.is_fake ? 'FAKE DETECTED' : 'AUTHENTIC MEDIA'}
                                </span>
                                <span className="font-mono font-bold text-white">
                                    {selectedItem.result.confidence_score}% Confidence
                                </span>
                            </div>
                            <div className="w-full bg-black/50 h-2 rounded-full overflow-hidden">
                                <div
                                    className={`h-full ${selectedItem.result.is_fake ? 'bg-neon-red' : 'bg-neon-green'}`}
                                    style={{ width: `${selectedItem.result.confidence_score}%` }}
                                ></div>
                            </div>
                        </div>

                        {selectedItem.result.breakdown && (
                            <div>
                                <h4 className="font-bold mb-3 text-sm uppercase tracking-wider text-gray-400">Analysis Breakdown</h4>
                                <div className="grid grid-cols-2 gap-3">
                                    {Object.entries(selectedItem.result.breakdown).map(([key, value]) => (
                                        <div key={key} className="flex justify-between items-center p-3 bg-black/20 rounded border border-gray-800">
                                            <span className="text-sm text-gray-300 capitalize">{key.replace(/_/g, ' ')}</span>
                                            <span className="font-mono font-bold text-neon-blue">{value}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {selectedItem.result.message && (
                            <div className="mt-6 p-3 bg-blue-900/20 border border-blue-500/30 rounded text-sm text-blue-200">
                                {selectedItem.result.message}
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default History;
