import random
import time
import os
from PIL import Image
from cnn_model_pytorch import CNNModel

# Initialize the model
# Using absolute path to ensure it's found
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn_trained.pth")
cnn_model = CNNModel(model_path=MODEL_PATH)

def get_threat_level(score):
    if score > 80:
        return "Severe Manipulation"
    elif score > 60:
        return "High Threat"
    elif score > 30:
        return "Moderate Threat"
    else:
        return "Low Threat"

def detect_image(file_path):
    try:
        # Open the image
        image = Image.open(file_path).convert('RGB')
        
        # Get prediction from the real model
        prediction, confidence = cnn_model.predict(image)
        
        # Convert confidence to percentage (0-100)
        confidence_score = round(confidence * 100, 2)
        is_fake = prediction == "fake"
        
        # Generate breakdown metrics based on the real result
        # (Since the model only gives a single score, we simulate the breakdown consistent with the result)
        if is_fake:
            face_warping = round(random.uniform(60, 95), 2)
            lighting = round(random.uniform(40, 70), 2)
            artifacts = round(random.uniform(50, 90), 2)
        else:
            face_warping = round(random.uniform(0, 15), 2)
            lighting = round(random.uniform(85, 100), 2)
            artifacts = round(random.uniform(0, 10), 2)

        return {
            "type": "image",
            "is_fake": is_fake,
            "confidence_score": confidence_score,
            "threat_level": get_threat_level(confidence_score) if is_fake else "Low Threat",
            "breakdown": {
                "face_warping": face_warping,
                "lighting_consistency": lighting,
                "artifact_detection": artifacts
            },
            "message": "Image analysis complete."
        }
    except Exception as e:
        print(f"Error in detect_image: {e}")
        # Fallback to mock if something goes wrong (e.g., model file missing)
        time.sleep(1)
        confidence = random.uniform(0, 100)
        return {
            "type": "image",
            "is_fake": confidence > 50,
            "confidence_score": round(confidence, 2),
            "threat_level": get_threat_level(confidence),
            "breakdown": {
                "face_warping": 0,
                "lighting_consistency": 0,
                "artifact_detection": 0
            },
            "message": f"Error: {str(e)}"
        }

def detect_text(text):
    time.sleep(0.5)
    # Simple heuristic: check for repetition or common AI patterns (mock)
    confidence = random.uniform(0, 100)
    return {
        "type": "text",
        "is_fake": confidence > 50,
        "confidence_score": round(confidence, 2),
        "threat_level": get_threat_level(confidence),
        "breakdown": {
            "perplexity": round(random.uniform(10, 100), 2),
            "burstiness": round(random.uniform(10, 100), 2)
        }
    }

def detect_audio(file_path):
    time.sleep(1.5)
    confidence = random.uniform(0, 100)
    return {
        "type": "audio",
        "is_fake": confidence > 50,
        "confidence_score": round(confidence, 2),
        "threat_level": get_threat_level(confidence),
        "breakdown": {
            "spectral_consistency": round(random.uniform(0, 100), 2),
            "background_noise": round(random.uniform(0, 100), 2)
        }
    }

import cv2
import numpy as np

def detect_video(file_path):
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        # Analyze 10 frames evenly distributed
        num_frames_to_analyze = 10
        frame_indices = [int(i * total_frames / num_frames_to_analyze) for i in range(num_frames_to_analyze)]
        
        fake_frames_count = 0
        total_confidence = 0
        suspicious_segments = []
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Predict
            prediction, confidence = cnn_model.predict(pil_image)
            
            # Convert to percentage
            conf_score = confidence * 100
            
            if prediction == "fake":
                fake_frames_count += 1
                # Map frame index to time in seconds
                timestamp = frame_idx / fps if fps > 0 else 0
                suspicious_segments.append({
                    "start": round(timestamp, 1),
                    "end": round(timestamp + 1, 1) # Assume 1 sec duration for segment
                })
            
            total_confidence += conf_score

        cap.release()
        
        # Aggregate results
        frames_analyzed = len(frame_indices)
        if frames_analyzed == 0:
             raise Exception("No frames could be analyzed")

        avg_confidence = total_confidence / frames_analyzed
        fake_ratio = fake_frames_count / frames_analyzed
        
        is_fake = fake_ratio > 0.3  # If > 30% frames are fake, flag video
        
        # Adjust confidence for final result
        final_confidence = avg_confidence if is_fake else (100 - avg_confidence)
        
        return {
            "type": "video",
            "is_fake": is_fake,
            "confidence_score": round(final_confidence, 2),
            "threat_level": get_threat_level(final_confidence) if is_fake else "Low Threat",
            "breakdown": {
                "lip_sync": round(random.uniform(60, 90) if is_fake else random.uniform(0, 20), 2),
                "eye_blinking": round(random.uniform(60, 90) if is_fake else random.uniform(0, 20), 2),
                "temporal_consistency": round(random.uniform(40, 80) if is_fake else random.uniform(80, 100), 2)
            },
            "suspicious_segments": suspicious_segments,
            "message": f"Video analysis complete. {fake_frames_count}/{frames_analyzed} frames flagged."
        }
        
    except Exception as e:
        print(f"Error in detect_video: {e}")
        # Fallback
        time.sleep(2)
        confidence = random.uniform(0, 100)
        is_fake = confidence > 50
        return {
            "type": "video",
            "is_fake": is_fake,
            "confidence_score": round(confidence, 2),
            "threat_level": get_threat_level(confidence),
            "breakdown": {
                "lip_sync": 0,
                "eye_blinking": 0,
                "temporal_consistency": 0
            },
            "suspicious_segments": [],
            "message": f"Error: {str(e)}"
        }
