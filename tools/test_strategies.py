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
        print(f'\n=== {car_name} ===')
        enhanced = recognizer._preprocess(plate_img)
        print(f'Enhanced: mean={enhanced.mean():.1f} shape={enhanced.shape}')

        # Strategy 1
        text1 = recognizer._recognize_full_readtext(enhanced)
        print(f'Strategy 1 (readtext): {repr(text1)} (len={len(text1)})')

        # Strategy 2
        text2 = recognizer._recognize_grid(enhanced)
        print(f'Strategy 2 (grid): {repr(text2)} (len={len(text2)})')

        # Strategy 3
        text3 = recognizer._recognize_full_crnn(enhanced)
        print(f'Strategy 3 (full CRNN): {repr(text3)} (len={len(text3)})')

        # Plausibility checks
        print(f'  is_plausible(S1): {recognizer._is_plausible(text1)}')
        print(f'  is_plausible(S2): {recognizer._is_plausible(text2)}')
        print(f'  is_plausible(S3): {recognizer._is_plausible(text3)}')
