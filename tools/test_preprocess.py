import sys, os, cv2
sys.path.insert(0, 'D:\\University\\PythonProject2')
from plate_recognition.detector import PlateDetector
import numpy as np

detector = PlateDetector()

for car_name in ['car.jpg', 'car2.jpg']:
    path = os.path.join('D:\\University\\PythonProject2', car_name)
    img = cv2.imread(path)
    candidates = detector.detect(img)

    for bi, (bbox, plate_img) in enumerate(candidates[:1]):
        print(f'\n=== {car_name} ===')
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        print(f'Original gray: mean={gray.mean():.1f} min={gray.min()} max={gray.max()}')
        print(f'Histogram: {np.histogram(gray, bins=5)[0]}')

        # Try different enhancements
        # 1. CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        clahe1 = clahe.apply(gray)
        print(f'CLAHE: mean={clahe1.mean():.1f} min={clahe1.min()} max={clahe1.max()}')

        # 2. Normalize
        norm1 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        print(f'Normalize: mean={norm1.mean():.1f} min={norm1.min()} max={norm1.max()}')

        # 3. CLAHE + Normalize + Invert
        gray2 = 255 - norm1
        clahe2 = clahe.apply(gray2)
        print(f'Invert+Normalize+CLAHE: mean={clahe2.mean():.1f} min={clahe2.min()} max={clahe2.max()}')

        # 4. EqualizeHist
        eq = cv2.equalizeHist(gray)
        print(f'EqualizeHist: mean={eq.mean():.1f} min={eq.min()} max={eq.max()}')

        # Try EasyOCR readtext with each variant
        import easyocr
        reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        for label, enhanced in [('CLAHE', clahe1), ('Normalize', norm1), ('Inv+Norm+CLAHE', clahe2), ('EqualizeHist', eq)]:
            h, w = enhanced.shape
            scale = max(1.0, 400.0 / h)
            up = cv2.resize(enhanced, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            rgb = cv2.cvtColor(up, cv2.COLOR_GRAY2RGB)
            results = reader.readtext(rgb, text_threshold=0.3, width_ths=0.1, add_margin=0.1)
            if results:
                entries = sorted([((b[0][0]+b[2][0])/2, t, c) for b,t,c in results], key=lambda e: e[0])
                full = "".join(t for _, t, _ in entries)
                print(f'  {label}: {len(results)} boxes, text={repr(full)}')
            else:
                print(f'  {label}: no text detected')
