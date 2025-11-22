# DeepTruth AI - Deep Fake Detection System

## Problem Statement
In an era of rapidly advancing generative AI, the proliferation of deepfakes poses a significant threat to information integrity, personal privacy, and social stability. Malicious actors can easily create realistic fake images, videos, and audio to spread misinformation, commit fraud, or harass individuals. There is an urgent need for accessible, accurate, and multi-modal detection tools to help users verify the authenticity of digital media.

## Objectives
1.  **Multi-Modal Detection**: Develop a system capable of analyzing Text, Images, Audio, and Video for signs of manipulation.
2.  **High Accuracy**: Integrate state-of-the-art CNN models for image detection to achieve high precision.
3.  **User-Centric Design**: Create an intuitive, "Cyber-Forensic" themed interface that makes complex forensic analysis accessible to non-experts.
4.  **Transparency**: Provide "Explain My Score" breakdowns to help users understand *why* content was flagged.
5.  **Verification**: Enable users to generate and download "Certificates of Authenticity" for verified media.

## Technology Stack
*   **Frontend**: React.js, Vite, Tailwind CSS v4 (Cyber-Forensic Theme)
*   **Backend**: Python, FastAPI, Uvicorn
*   **AI/ML**: PyTorch (Custom CNN), Librosa, Pillow, NumPy
*   **Audio Detection**: CNN-based spectrogram analysis (based on Kaggle kernel: hakim11/deep-fake-voice-recognition-using-cnn)
*   **Database**: SQLite (History & Feedback tracking)

## Features
*   **Real-time Analysis**: Instant detection results for all modalities.
*   **Deepfake Rewind**: (Simulated) Video timeline analysis to pinpoint manipulated frames.
*   **Threat Level Assessment**: Categorizes risks from "Low" to "Severe".
*   **History Dashboard**: Track all past scans and view aggregate statistics.
*   **Authenticity Certificates**: Downloadable proof for verified content.

## Deployment Instructions (Railway)

### Prerequisites
*   GitHub account
*   Railway account (sign up at [railway.app](https://railway.app))

### Steps

1. **Push your code to GitHub** (already done!)

2. **Create a new project on Railway**:
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your `deepfake-` repository

3. **Railway will auto-detect both services**:
   - `backend` (Python/FastAPI)
   - `frontend` (React/Vite)

4. **Configure Environment Variables**:
   - Click on the **Frontend** service
   - Go to "Variables" tab
   - Add: `VITE_API_URL` = `https://[your-backend-service-url]`
   - (Railway will show you the backend URL after deployment)

5. **Deploy**:
   - Railway will automatically build and deploy both services
   - Wait for deployment to complete (~3-5 minutes)

6. **Get your URLs**:
   - Backend: `https://deepfake-detection-backend.up.railway.app`
   - Frontend: `https://deepfake-detection-frontend.up.railway.app`

### Audio Model Setup

The audio deepfake detection uses a CNN-based model that analyzes mel-spectrograms. You have two options:

#### Option 1: Download Pre-trained Model from Kaggle (Recommended)

```bash
cd backend
python download_kaggle_model.py
```

Or manually using Kaggle CLI:
```bash
pip install kaggle
kaggle kernels output hakim11/deep-fake-voice-recognition-using-cnn -p ./backend
# Rename the downloaded model file to audio_cnn_trained.pth if needed
```

#### Option 2: Train Your Own Model

If you have a dataset with real and fake audio samples:

```bash
cd backend
python train_audio_model.py \
    --train_dir /path/to/train \
    --val_dir /path/to/val \
    --epochs 50 \
    --batch_size 32
```

Expected directory structure:
```
train/
  real/
    audio1.wav
    audio2.wav
    ...
  fake/
    audio1.wav
    audio2.wav
    ...
val/
  real/
    audio1.wav
    ...
  fake/
    audio1.wav
    ...
```

**Note**: If no trained model is available, the system will use fallback spectral analysis.

### Alternative: Local Deployment

If you want to run locally instead:

```bash
# Backend
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

## Usage
1.  Open the frontend URL in your browser.
2.  Navigate to the **Detector** tab.
3.  Select the media type (Image, Video, Audio, Text).
4.  Upload a file or paste text.
5.  Click **INITIATE SCAN**.
6.  View the Threat Level, Confidence Score, and Breakdown.
7.  If authentic, click **DOWNLOAD CERTIFICATE**.
