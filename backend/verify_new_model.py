import os
import torch
import librosa
import torch.nn.functional as F
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

AUDIO_MODEL_NAME = "mo-thecreator/Deepfake-audio-detection"
TEST_FILE = "test.wav"

def verify_model():
    print(f"Loading model: {AUDIO_MODEL_NAME}...")
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(AUDIO_MODEL_NAME)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FAILED to load model: {e}")
        return

    if not os.path.exists(TEST_FILE):
        print(f"Test file {TEST_FILE} not found. Creating a dummy one...")
        import soundfile as sf
        import numpy as np
        sr = 16000
        dummy_audio = np.random.uniform(-1, 1, sr * 3) # 3 seconds of noise
        sf.write(TEST_FILE, dummy_audio, sr)
        print(f"Created {TEST_FILE}")

    print(f"Processing {TEST_FILE}...")
    try:
        audio, sample_rate = librosa.load(TEST_FILE, sr=16000)
        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = model(**inputs).logits
        
        probs = F.softmax(logits, dim=-1)
        print(f"Logits: {logits}")
        print(f"Probabilities: {probs}")
        
        # Check label mapping
        print(f"Model config id2label: {model.config.id2label}")
        
        fake_prob = probs[0][1].item()
        real_prob = probs[0][0].item()
        
        print(f"Real Probability: {real_prob:.4f}")
        print(f"Fake Probability: {fake_prob:.4f}")
        
        if fake_prob > real_prob:
            print("Prediction: FAKE")
        else:
            print("Prediction: REAL")
            
    except Exception as e:
        print(f"FAILED during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_model()
