"""Test dynamic dot finder on car.jpg and car2.jpg."""
import cv2, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from plate_recognition.detector import PlateDetector
from plate_recognition.recognizer import CharRecognizer, CHAR_MAP, _get_allowed_indices

detector = PlateDetector()
recognizer = CharRecognizer()
recognizer.load_models()

for car_name in ['car.jpg', 'car2.jpg']:
    print(f'===== {car_name} =====')
    img = cv2.imread(car_name)
    candidates = detector.detect(img)
    for bbox, plate_img in candidates[:1]:
        base = recognizer.preprocess(plate_img)
        h, w = base.shape
        dot_x = recognizer._find_dot(base)
        print(f'Dot at x={dot_x} ({dot_x/w*100:.1f}% of {w})')
        boxes = recognizer._build_slot_boxes(h, w, base)
        print(f'Boxes: {len(boxes)}')
        for pos, (x1, x2, y1, y2) in enumerate(boxes):
            slot = base[y1:y2, x1:x2]
            processed = recognizer.preprocess_char_image(slot)
            tensor = torch.from_numpy(processed.astype(np.float32) / 255.0)
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = recognizer._cnn_model(tensor)
                probs = F.softmax(logits, dim=1)[0]
            top3_idx = torch.topk(probs, 3).indices.tolist()
            top3_prob = torch.topk(probs, 3).values.tolist()
            allowed = _get_allowed_indices(pos)
            mask = torch.zeros(67)
            mask[allowed] = 1.0
            best = torch.argmax(probs * mask).item()
            best_char = CHAR_MAP.get(best, '?')
            best_conf = (probs * mask)[best].item()
            top_strs = []
            for i, p in zip(top3_idx, top3_prob):
                ch = CHAR_MAP.get(i, '?')
                ok = 'OK' if i in allowed else 'X'
                top_strs.append(f'{ch}:{p:.2f}({ok})')
            print(f'  Slot {pos} x=[{x1},{x2}]: {", ".join(top_strs)} => {ascii(best_char)}:{best_conf:.3f}')
        result = recognizer.recognize_plate(plate_img)
        actual = '京BK3551' if car_name == 'car.jpg' else '粤R458C'
        print(f'Result: {ascii(result)}')
        print(f'Actual: {ascii(actual)}')
        print()
