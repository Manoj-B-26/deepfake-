"""Test audio detection directly"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models import detect_audio

# Test with test.wav if it exists
test_file = 'test.wav'
if not os.path.exists(test_file):
    print(f"Test file {test_file} not found")
    sys.exit(1)

print(f"Testing audio detection on {test_file}...")
try:
    result = detect_audio(test_file)
    print("\nResult:")
    import json
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

