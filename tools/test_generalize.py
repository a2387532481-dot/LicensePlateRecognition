"""Test generalization on car3.jpg and car4.jpg."""
import cv2, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from plate_recognition.detector import PlateDetector
from plate_recognition.recognizer import CharRecognizer, CHAR_MAP, _get_allowed_indices

detector = PlateDetector()
recognizer = CharRecognizer()
recognizer.load_models()

for car_name in ['car3.jpg', 'car4.jpg']:
    print(f'===== {car_name} =====')
    img = cv2.imread(car_name)
    if img is None:
        print(f'  ERROR: cannot read {car_name}')
        continue
    print(f'  Image: {img.shape}')
    candidates = detector.detect(img)
    print(f'  Candidates: {len(candidates)}')
    if not candidates:
        print(f'  WARNING: no plate detected')
        continue
    for bi, (bbox, plate_img) in enumerate(candidates[:2]):
        base = recognizer.preprocess(plate_img)
        h, w = base.shape
        dot_x = recognizer._find_dot(base)
        n_total = recognizer._estimate_char_count(base)
        print(f'  Candidate {bi}: plate={plate_img.shape}, pre_mean={base.mean():.1f}')
        print(f'    dot_x={dot_x} ({dot_x/w*100:.1f}%), estimated_chars={n_total}')
        boxes = recognizer._build_slot_boxes(h, w, base)
        for pos, (x1, x2, y1, y2) in enumerate(boxes):
            slot = base[y1:y2, x1:x2]
            processed = recognizer.preprocess_char_image(slot)
            tensor = torch.from_numpy(processed.astype(np.float32) / 255.0)
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = recognizer._cnn_model(tensor)
                probs = F.softmax(logits, dim=1)[0]
            allowed = _get_allowed_indices(pos)
            mask = torch.zeros(67)
            mask[allowed] = 1.0
            best = torch.argmax(probs * mask).item()
            best_char = CHAR_MAP.get(best, '?')
            best_conf = (probs * mask)[best].item()
            top3_idx = torch.topk(probs, 3).indices.tolist()
            top3_prob = torch.topk(probs, 3).values.tolist()
            top_strs = []
            for idx, prob in zip(top3_idx, top3_prob):
                ch = CHAR_MAP.get(idx, '?')
                ok = 'OK' if idx in allowed else 'X'
                top_strs.append(f'{ch}:{prob:.2f}({ok})')
            print(f'    Slot {pos}: {ascii(best_char):6s} conf={best_conf:.3f}  top3=[{", ".join(top_strs)}]')
        result = recognizer.recognize_plate(plate_img)
        print(f'    Result: {ascii(result)}')
    print()
