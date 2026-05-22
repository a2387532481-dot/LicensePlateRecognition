import sys, os, cv2
sys.path.insert(0, 'D:\\University\\PythonProject2')
from plate_recognition.recognizer import CharRecognizer
from plate_recognition.detector import PlateDetector

recognizer = CharRecognizer(device='cpu')
recognizer.load_models()
detector = PlateDetector()

for car_name in ['car.jpg', 'car2.jpg']:
    path = os.path.join('D:\\University\\PythonProject2', car_name)
    img = cv2.imread(path)
    candidates = detector.detect(img)

    for bi, (bbox, plate_img) in enumerate(candidates[:1]):
        print(f'\n=== {car_name} plate {bi} ===')
        enhanced = recognizer._preprocess(plate_img)
        h, w = enhanced.shape

        # Upscale for better detection
        scale = max(1.0, 400.0 / h)
        up = cv2.resize(enhanced, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(up, cv2.COLOR_GRAY2RGB)

        for width_ths in [0.5, 0.1, 0.05, 0.01]:
            for text_th in [0.7, 0.5, 0.3, 0.1]:
                results = recognizer._reader.readtext(
                    rgb, text_threshold=text_th, width_ths=width_ths,
                    add_margin=0.1,
                )
                entries = sorted(
                    [((b[0][0] + b[2][0]) / 2, t, c) for b, t, c in results],
                    key=lambda e: e[0],
                )
                full = "".join(t for _, t, _ in entries)
                if len(full) >= 5:
                    print(f'  wths={width_ths:.2f} tth={text_th:.1f}: {len(entries)} detections: {repr(full)}')
                    break
