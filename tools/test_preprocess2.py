import sys, os, cv2
sys.path.insert(0, 'D:\\University\\PythonProject2')
from plate_recognition.detector import PlateDetector
import easyocr
import numpy as np

detector = PlateDetector()
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

for car_name in ['car.jpg']:
    path = os.path.join('D:\\University\\PythonProject2', car_name)
    img = cv2.imread(path)
    candidates = detector.detect(img)

    for bi, (bbox, plate_img) in enumerate(candidates[:1]):
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # Aggressive preprocessing
        # 1. Normalize to full range
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # 2. Invert (white text on dark bg → dark text on light bg)
        inv = 255 - norm

        # 3. Strong CLAHE
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(inv)

        # 4. Sharpen
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
        enhanced = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

        print(f'Enhanced: mean={enhanced.mean():.1f} min={enhanced.min()} max={enhanced.max()}')

        h, w = enhanced.shape
        scale = max(1.0, 400.0 / h)
        up = cv2.resize(enhanced, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(up, cv2.COLOR_GRAY2RGB)

        for th in [0.7, 0.5, 0.3, 0.1, 0.05]:
            results = reader.readtext(rgb, text_threshold=th, width_ths=0.1, add_margin=0.1)
            if results:
                entries = sorted([((b[0][0]+b[2][0])/2, t, c) for b,t,c in results], key=lambda e: e[0])
                full = "".join(t for _, t, _ in entries)
                print(f'  th={th}: {len(results)} boxes: {repr(full)}')
                break
        else:
            print('  No text detected with any threshold')

        # Also try EasyOCR readtext on original plate (not preprocessed)
        plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        h2, w2 = plate_rgb.shape[:2]
        sc2 = max(1.0, 400.0 / h2)
        plate_up = cv2.resize(plate_rgb, (int(w2*sc2), int(h2*sc2)))
        for th in [0.7, 0.5, 0.3, 0.1, 0.05]:
            results = reader.readtext(plate_up, text_threshold=th, width_ths=0.1, add_margin=0.1)
            if results:
                entries = sorted([((b[0][0]+b[2][0])/2, t, c) for b,t,c in results], key=lambda e: e[0])
                full = "".join(t for _, t, _ in entries)
                print(f'  Original RGB th={th}: {len(results)} boxes: {repr(full)}')
                break
        else:
            print('  Original RGB: No text detected')
