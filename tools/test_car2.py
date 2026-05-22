import sys, os, cv2
sys.path.insert(0, 'D:\\University\\PythonProject2')
from plate_recognition.detector import PlateDetector
from plate_recognition.recognizer import CharRecognizer
import numpy as np

detector = PlateDetector()
recognizer = CharRecognizer(device='cpu')
recognizer.load_models()

path = os.path.join('D:\\University\\PythonProject2', 'car2.jpg')
img = cv2.imread(path)
candidates = detector.detect(img)

for bi, (bbox, plate_img) in enumerate(candidates[:1]):
    enhanced = recognizer._preprocess(plate_img)
    print(f'Enhanced: mean={enhanced.mean():.1f} shape={enhanced.shape}')

    h, w = enhanced.shape
    scale = max(1.0, 400.0 / h)
    up = cv2.resize(enhanced, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(up, cv2.COLOR_GRAY2RGB)

    for wths in [0.5, 0.1, 0.05, 0.01]:
        for tth in [0.7, 0.5, 0.3, 0.1]:
            results = recognizer._reader.readtext(
                rgb, text_threshold=tth, width_ths=wths, add_margin=0.15,
            )
            if results:
                for b, t, c in results:
                    codepoints = ' '.join(f'U+{ord(ch):04X}' for ch in t)
                    print(f'  wths={wths} tth={tth}: {repr(t)} ({codepoints}) conf={c:.3f}')
                break
        else:
            print(f'  wths={wths}: no text detected at any threshold')
