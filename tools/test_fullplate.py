import sys, os, cv2
sys.path.insert(0, 'D:\\University\\PythonProject2')
from plate_recognition.recognizer import CharRecognizer
from plate_recognition.detector import PlateDetector
import numpy as np

recognizer = CharRecognizer(device='cpu')
recognizer.load_models()
detector = PlateDetector()

for car_name in ['car.jpg', 'car2.jpg']:
    path = os.path.join('D:\\University\\PythonProject2', car_name)
    img = cv2.imread(path)
    candidates = detector.detect(img)

    for bi, (bbox, plate_img) in enumerate(candidates[:1]):
        print(f'\n=== {car_name} ===')

        # Full plate recognize() — CRNN reads entire plate as one text line
        enhanced = recognizer._preprocess(plate_img)
        h, w = enhanced.shape
        print(f'Enhanced: mean={enhanced.mean():.1f}')

        # Try different upscale sizes
        for scale in [1.0, 2.0, 3.0]:
            new_w = int(w * scale)
            new_h = int(h * scale)
            up = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            rgb = cv2.cvtColor(up, cv2.COLOR_GRAY2RGB)

            results = recognizer._reader.recognize(
                rgb,
                horizontal_list=[[0, new_w, 0, new_h]],
                free_list=[],
                detail=1,
            )
            if results:
                _, text, conf = results[0]
                print(f'  scale={scale:.0f}x: {repr(text)} (conf={conf:.3f})')

        # Also try grid-based per-slot
        print('  Grid per-slot:')
        slot_results = recognizer._grid_recognize(enhanced)
        for i, (char, conf) in enumerate(slot_results):
            codepoints = ' '.join(f'U+{ord(c):04X}' for c in char) if char else ''
            print(f'    Slot {i}: {repr(char)} ({codepoints}) conf={conf:.3f}')
        assembled = recognizer._assemble(slot_results)
        print(f'    Assembled: {repr(assembled)}')
