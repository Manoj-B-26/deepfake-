import sys
import os

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import detect_audio

print("Testing detect_audio with real_noisy.wav...")
try:
    result = detect_audio('real_noisy.wav')
    print("Result:", result)
    with open('result.txt', 'w') as f:
        f.write(str(result))
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()
