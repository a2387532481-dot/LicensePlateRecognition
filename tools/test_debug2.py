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
        print(f'\n=== {car_name} plate {bi} (shape={plate_img.shape}) ===')

        enhanced = recognizer._preprocess(plate_img)
        print(f'Enhanced: mean={enhanced.mean():.1f}, min={enhanced.min()}, max={enhanced.max()}')

        # Check polarity
        h, w = enhanced.shape
        center_mean = enhanced[h//3:2*h//3, w//4:3*w//4].mean()
        print(f'Center mean: {center_mean:.1f}')

        slot_results = recognizer._grid_recognize(enhanced)

        for i, (char, conf) in enumerate(slot_results):
            codepoints = ' '.join(f'U+{ord(c):04X}' for c in char)
            print(f'  Slot {i}: char={repr(char)} (len={len(char)}, codepoints={codepoints}) conf={conf:.3f}')

        text = recognizer._assemble(slot_results)
        print(f'  Assembled: {repr(text)} (len={len(text)})')

        text = recognizer._validate_and_retry(enhanced, text)
        print(f'  After retry: {repr(text)} (len={len(text)})')
