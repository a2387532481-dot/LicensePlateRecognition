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
        print(f'\n=== {car_name} plate {bi} (shape={plate_img.shape}) ===')

        # Save original plate
        cv2.imwrite(f'debug_plate_{car_name}_{bi}.jpg', plate_img)

        enhanced = recognizer._preprocess(plate_img)
        cv2.imwrite(f'debug_enhanced_{car_name}_{bi}.jpg', enhanced)

        h, w = enhanced.shape

        # Replicate _grid_recognize slot extraction
        dot_x = recognizer._find_dot(enhanced)
        if dot_x is None:
            dot_x = int(w * 0.285)

        left_margin = int(w * 0.015)
        right_margin = int(w * 0.01)
        dot_half = int(h * 0.10)
        left_end = dot_x - dot_half
        right_start = dot_x + dot_half
        left_w = left_end - left_margin
        right_w = w - right_start - right_margin
        left_slot_w = left_w / 2
        right_slot_w = right_w / 5

        slot_centers = []
        for i in range(2):
            slot_centers.append(left_margin + int(left_slot_w * (i + 0.5)))
        for i in range(5):
            slot_centers.append(right_start + int(right_slot_w * (i + 0.5)))

        for i, cx in enumerate(slot_centers):
            if i < 2:
                hw = max(int(left_slot_w * 0.45), 10)
            else:
                hw = max(int(right_slot_w * 0.45), 10)
            x0 = max(0, cx - hw)
            x1 = min(w, cx + hw)
            slot_img = enhanced[0:h, x0:x1]

            # Save raw slot
            cv2.imwrite(f'debug_slot_{car_name}_{bi}_{i}_raw.jpg', slot_img)

            # What _ocr_slot does to it
            _, binary = cv2.threshold(slot_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if binary.mean() > 128:
                text_mask = 255 - binary
            else:
                text_mask = binary

            cv2.imwrite(f'debug_slot_{car_name}_{bi}_{i}_binary.jpg', text_mask)

            if text_mask.mean() < 3:
                continue

            target_h = 90
            scale = target_h / max(slot_img.shape[0], 1)
            target_w = max(20, int(slot_img.shape[1] * scale))
            scaled = cv2.resize(text_mask, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            _, scaled = cv2.threshold(scaled, 127, 255, cv2.THRESH_BINARY)

            pad = 20
            canvas_size = max(target_h, target_w) + 2 * pad
            canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
            y_off = (canvas_size - target_h) // 2
            x_off = (canvas_size - target_w) // 2
            canvas[y_off:y_off + target_h, x_off:x_off + target_w] = scaled

            # Invert for EasyOCR (dark text on light bg)
            canvas = 255 - canvas
            cv2.imwrite(f'debug_slot_{car_name}_{bi}_{i}_ocr_input.jpg', canvas)

        print(f'Saved debug images for {car_name}')
