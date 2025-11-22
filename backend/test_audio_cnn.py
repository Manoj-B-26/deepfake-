"""
Quick test script to verify Audio CNN model setup
"""

import os
import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        from audio_cnn_model import AudioCNNModel
        print("✓ AudioCNNModel imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import AudioCNNModel: {e}")
        return False

def test_model_initialization():
    """Test if model can be initialized"""
    print("\nTesting model initialization...")
    try:
        from audio_cnn_model import AudioCNNModel
        model = AudioCNNModel(model_path="audio_cnn_trained.pth")
        print("✓ Model initialized successfully")
        
        # Try to build the model
        model.build_model()
        print("✓ Model architecture built successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        return False

def test_model_file_exists():
    """Check if trained model file exists"""
    print("\nChecking for trained model file...")
    model_path = "audio_cnn_trained.pth"
    if os.path.exists(model_path):
        print(f"✓ Found trained model: {model_path}")
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        print(f"  File size: {file_size:.2f} MB")
        return True
    else:
        print(f"✗ Trained model not found: {model_path}")
        print("  Run download_kaggle_model.py or train_audio_model.py to get a trained model")
        return False

def test_audio_processing():
    """Test if audio can be processed (if test.wav exists)"""
    print("\nTesting audio processing...")
    test_audio = "test.wav"
    
    if not os.path.exists(test_audio):
        print(f"⚠ Test audio file not found: {test_audio}")
        print("  Skipping audio processing test")
        return True
    
    try:
        import librosa
        audio, sr = librosa.load(test_audio, sr=22050, duration=3.0)
        print(f"✓ Audio loaded successfully")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Duration: {len(audio)/sr:.2f} seconds")
        print(f"  Samples: {len(audio)}")
        
        # Test spectrogram generation
        from audio_cnn_model import AudioCNNModel
        model = AudioCNNModel(model_path="audio_cnn_trained.pth")
        spec = model.audio_to_spectrogram(test_audio)
        print(f"✓ Spectrogram generated successfully")
        print(f"  Shape: {spec.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Audio processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*70)
    print("Audio CNN Model Test")
    print("="*70)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Model Initialization", test_model_initialization()))
    results.append(("Model File", test_model_file_exists()))
    results.append(("Audio Processing", test_audio_processing()))
    
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL/SKIP"
        print(f"{test_name:30s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
        print("\nNext steps:")
        print("  1. If model file doesn't exist, run: python download_kaggle_model.py")
        print("  2. Or train your own model: python train_audio_model.py")
        print("  3. Test audio detection: python test_audio.py")
    else:
        print("\n⚠ Some tests failed or were skipped")
        print("\nTroubleshooting:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Download model from Kaggle: python download_kaggle_model.py")
        print("  3. Or train your own model: python train_audio_model.py --help")
    
    print("="*70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

