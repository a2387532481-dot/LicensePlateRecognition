"""Debug: show top-5 CNN predictions for each slot."""
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
        boxes = recognizer._build_slot_boxes(h, w)

        for pos in range(min(7, len(boxes))):
            x1, x2, y1, y2 = boxes[pos]
            slot = base[y1:y2, x1:x2]
            processed = recognizer.preprocess_char_image(slot)

            tensor = torch.from_numpy(processed.astype(np.float32) / 255.0)
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = recognizer._cnn_model(tensor)
                probs = F.softmax(logits, dim=1)[0]

            top5_idx = torch.topk(probs, 5).indices.tolist()
            top5_prob = torch.topk(probs, 5).values.tolist()
            print(f'Slot {pos} ({slot.shape[1]}x{slot.shape[0]}):')
            for i, (idx, prob) in enumerate(zip(top5_idx, top5_prob)):
                char = CHAR_MAP.get(idx, '?')
                allowed = idx in _get_allowed_indices(pos)
                mark = 'OK' if allowed else 'BLOCKED'
                print(f'  Top{i+1}: idx={idx:2d} char={ascii(char):6s} prob={prob:.4f} {mark}')
            allowed = _get_allowed_indices(pos)
            mask = torch.zeros(67)
            mask[allowed] = 1.0
            masked = probs * mask
            best = torch.argmax(masked).item()
            print(f'  => Best allowed: idx={best} char={ascii(CHAR_MAP.get(best, "?"))} prob={masked[best].item():.4f}')
            print()
