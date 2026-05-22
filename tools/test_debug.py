import sys, os, cv2
sys.path.insert(0, 'D:\\University\\PythonProject2')
from plate_recognition.recognizer import CharRecognizer
from plate_recognition.detector import PlateDetector
import numpy as np

recognizer = CharRecognizer(device='cpu')
recognizer.load_models()
detector = PlateDetector()
print('Loaded.')

for car_name in ['car.jpg', 'car2.jpg']:
    path = os.path.join('D:\\University\\PythonProject2', car_name)
    if not os.path.isfile(path):
        continue
    img = cv2.imread(path)
    candidates = detector.detect(img)
    if not candidates:
        print(f'{car_name}: no plates detected')
        continue

    for bi, (bbox, plate_img) in enumerate(candidates):
        print(f'\n=== {car_name} plate {bi} (shape={plate_img.shape}) ===')

        enhanced = recognizer._preprocess(plate_img)
        slot_results = recognizer._grid_recognize(enhanced)

        for i, (char, conf) in enumerate(slot_results):
            print(f'  Slot {i}: char={repr(char)} conf={conf:.3f}')

        text = recognizer._assemble(slot_results)
        print(f'  Assembled: {repr(text)}')

        text = recognizer._validate_and_retry(enhanced, text)
        print(f'  After retry: {repr(text)}')

        text = recognizer._apply_plate_rules(text)
        print(f'  Final: {repr(text)}')
