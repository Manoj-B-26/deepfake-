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
*   **AI/ML**: PyTorch (Custom CNN), Pillow, NumPy
*   **Database**: SQLite (History & Feedback tracking)

## Features
*   **Real-time Analysis**: Instant detection results for all modalities.
*   **Deepfake Rewind**: (Simulated) Video timeline analysis to pinpoint manipulated frames.
*   **Threat Level Assessment**: Categorizes risks from "Low" to "Severe".
*   **History Dashboard**: Track all past scans and view aggregate statistics.
*   **Authenticity Certificates**: Downloadable proof for verified content.

## Deployment Instructions

### Prerequisites
*   Node.js & npm
*   Python 3.8+
*   pip

### 1. Backend Setup
Navigate to the backend directory and install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

Start the FastAPI server:
```bash
python -m uvicorn main:app --reload
```
The backend will run on `http://localhost:8000`.

### 2. Frontend Setup
Open a new terminal, navigate to the frontend directory, and install dependencies:
```bash
cd frontend
npm install
```

Start the development server:
```bash
npm run dev
```
The frontend will run on `http://localhost:5173`.

## Usage
1.  Open the frontend URL in your browser.
2.  Navigate to the **Detector** tab.
3.  Select the media type (Image, Video, Audio, Text).
4.  Upload a file or paste text.
5.  Click **INITIATE SCAN**.
6.  View the Threat Level, Confidence Score, and Breakdown.
7.  If authentic, click **DOWNLOAD CERTIFICATE**.
