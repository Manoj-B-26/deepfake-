import React, { useState } from 'react';
import { Upload, FileText, Image as ImageIcon, Mic, Video, X, CheckCircle, AlertTriangle, Loader2, Activity, ShieldCheck } from 'lucide-react';
import axios from 'axios';

const getApiUrl = () => {
    const url = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    return url.startsWith('http') ? url : `https://${url}`;
};
const API_URL = getApiUrl();

const Detector = () => {
    const [mode, setMode] = useState('image');
    const [file, setFile] = useState(null);
    const [textInput, setTextInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [preview, setPreview] = useState(null);

    const modes = [
        { id: 'image', label: 'Image Analysis', icon: ImageIcon },
        { id: 'video', label: 'Video Analysis', icon: Video },
        { id: 'audio', label: 'Audio Analysis', icon: Mic },
        { id: 'text', label: 'Text Analysis', icon: FileText },
    ];

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        console.log("File selected:", selectedFile);
        if (selectedFile) {
            setFile(selectedFile);
            if (mode === 'image') {
                const reader = new FileReader();
                reader.onloadend = () => setPreview(reader.result);
                reader.readAsDataURL(selectedFile);
            }
        }
    };

    const handleAnalyze = async () => {
        console.log("Analyze initiated. Mode:", mode);
        if (!file && !textInput) {
            console.warn("No file or text input provided.");
            return;
        }

        setLoading(true);
        setResult(null);

        try {
            let endpoint = `/detect/${mode}`;
            let data;

            console.log("Sending request to:", `${API_URL}${endpoint}`);

            if (mode === 'text') {
                const formData = new FormData();
                formData.append('text', textInput);
                data = formData;
            } else {
                const formData = new FormData();
                formData.append('file', file);
                data = formData;
            }

            const response = await axios.post(`${API_URL}${endpoint}`, data, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });

            console.log("Response received:", response.data);
            setResult(response.data);
        } catch (error) {
            console.error("Analysis failed:", error);
            alert(`Analysis failed: ${error.message}. Check console for details.`);
        } finally {
            setLoading(false);
        }
    };

    const getThreatColor = (level) => {
        switch (level) {
            case 'Severe Manipulation': return 'text-neon-red';
            case 'High Threat': return 'text-orange-500';
            case 'Moderate Threat': return 'text-yellow-400';
            default: return 'text-neon-green';
        }
    };

    return (
        <div className="max-w-6xl mx-auto animate-fade-in">
            <div className="flex space-x-4 mb-8 overflow-x-auto pb-2">
                {modes.map((m) => (
                    <button
                        key={m.id}
                        onClick={() => { setMode(m.id); setFile(null); setResult(null); setPreview(null); }}
                        className={`flex items-center space-x-2 px-6 py-3 rounded-lg border transition-all whitespace-nowrap ${mode === m.id
                            ? 'bg-neon-blue/10 border-neon-blue text-neon-blue shadow-[0_0_10px_rgba(0,243,255,0.2)]'
                            : 'bg-cyber-gray border-gray-800 text-gray-400 hover:border-gray-600'
                            }`}
                    >
                        <m.icon className="w-5 h-5" />
                        <span className="font-medium">{m.label}</span>
                    </button>
                ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Input Section */}
                <div className="bg-cyber-gray border border-gray-800 rounded-xl p-8">
                    <h2 className="text-2xl font-bold mb-6">Upload Source</h2>

                    {mode === 'text' ? (
                        <textarea
                            value={textInput}
                            onChange={(e) => setTextInput(e.target.value)}
                            className="w-full h-64 bg-black/30 border border-gray-700 rounded-lg p-4 text-gray-300 focus:border-neon-blue focus:outline-none transition-colors resize-none"
                            placeholder="Paste suspicious text here..."
                        />
                    ) : (
                        <div className="border-2 border-dashed border-gray-700 rounded-xl p-12 text-center hover:border-neon-blue/50 transition-colors relative group">
                            <input
                                type="file"
                                onChange={handleFileChange}
                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                accept={mode === 'image' ? 'image/*' : mode === 'video' ? 'video/*' : 'audio/*'}
                            />
                            {preview ? (
                                <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded-lg shadow-lg" />
                            ) : file ? (
                                <div className="text-neon-blue flex flex-col items-center">
                                    <CheckCircle className="w-12 h-12 mb-4" />
                                    <p className="font-mono">{file.name}</p>
                                </div>
                            ) : (
                                <div className="text-gray-500 flex flex-col items-center group-hover:text-gray-400 transition-colors">
                                    <Upload className="w-12 h-12 mb-4" />
                                    <p className="text-lg font-medium mb-2">Drop file here or click to upload</p>
                                    <p className="text-sm font-mono">Supports: {mode === 'image' ? 'JPG, PNG' : mode === 'video' ? 'MP4, MOV' : 'MP3, WAV'}</p>
                                </div>
                            )}
                        </div>
                    )}

                    <button
                        onClick={handleAnalyze}
                        disabled={loading || (!file && !textInput)}
                        className={`w-full mt-6 py-4 rounded-lg font-bold text-lg flex items-center justify-center space-x-2 transition-all ${loading || (!file && !textInput)
                            ? 'bg-gray-800 text-gray-500 cursor-not-allowed'
                            : 'bg-cyan-400 text-black hover:bg-cyan-300 shadow-neon-blue opacity-100'
                            }`}
                    >
                        {loading ? (
                            <>
                                <Loader2 className="w-6 h-6 animate-spin" />
                                <span>ANALYZING...</span>
                            </>
                        ) : (
                            <>
                                <ShieldCheck className="w-6 h-6" />
                                <span>INITIATE SCAN</span>
                            </>
                        )}
                    </button>
                </div>

                {/* Results Section */}
                <div className="bg-cyber-gray border border-gray-800 rounded-xl p-8 min-h-[500px] relative overflow-hidden">
                    {!result ? (
                        <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-600">
                            <Activity className="w-16 h-16 mb-4 opacity-20" />
                            <p className="font-mono">Awaiting Analysis Data...</p>
                        </div>
                    ) : (
                        <div className="animate-fade-in">
                            <div className="flex justify-between items-start mb-8">
                                <div>
                                    <h3 className="text-gray-400 font-mono text-sm mb-1">THREAT LEVEL</h3>
                                    <p className={`text-3xl font-bold ${getThreatColor(result.threat_level)}`}>
                                        {result.threat_level.toUpperCase()}
                                    </p>
                                </div>
                                <div className="text-right">
                                    <h3 className="text-gray-400 font-mono text-sm mb-1">CONFIDENCE</h3>
                                    <p className="text-3xl font-bold text-white">{result.confidence_score}%</p>
                                </div>
                            </div>

                            <div className="mb-8">
                                <h3 className="text-lg font-bold mb-4 flex items-center space-x-2">
                                    <Activity className="w-5 h-5 text-neon-blue" />
                                    <span>Explain My Score</span>
                                </h3>
                                <div className="grid grid-cols-1 gap-4">
                                    {Object.entries(result.breakdown).map(([key, value]) => (
                                        <div key={key} className="bg-black/30 p-4 rounded-lg border border-gray-800">
                                            <div className="flex justify-between mb-2">
                                                <span className="text-gray-300 capitalize">{key.replace('_', ' ')}</span>
                                                <span className={`font-mono ${value > 50 ? 'text-neon-red' : 'text-neon-green'}`}>{value}%</span>
                                            </div>
                                            <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                                                <div
                                                    className={`h-full ${value > 50 ? 'bg-neon-red' : 'bg-neon-green'}`}
                                                    style={{ width: `${value}%` }}
                                                ></div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {result.is_fake ? (
                                <div className="bg-neon-red/10 border border-neon-red/30 p-4 rounded-lg flex items-start space-x-3">
                                    <AlertTriangle className="w-6 h-6 text-neon-red flex-shrink-0" />
                                    <div>
                                        <h4 className="font-bold text-neon-red mb-1">Manipulation Detected</h4>
                                        <p className="text-sm text-gray-300">
                                            Our system has detected significant anomalies consistent with deepfake manipulation techniques.
                                        </p>
                                    </div>
                                </div>
                            ) : (
                                <div className="bg-neon-green/10 border border-neon-green/30 p-4 rounded-lg flex items-start space-x-3">
                                    <CheckCircle className="w-6 h-6 text-neon-green flex-shrink-0" />
                                    <div>
                                        <h4 className="font-bold text-neon-green mb-1">Verified Authentic</h4>
                                        <p className="text-sm text-gray-300">
                                            No significant signs of manipulation were found. This content appears to be authentic.
                                        </p>
                                        <button
                                            onClick={() => window.open(`${API_URL}/certificate/${result.scan_id}`, '_blank')}
                                            className="mt-3 text-xs bg-green-400 text-black px-3 py-1 rounded font-bold hover:bg-green-300 transition-colors opacity-100"
                                        >
                                            DOWNLOAD CERTIFICATE
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default Detector;
