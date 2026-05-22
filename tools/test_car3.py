import sys, os, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plate_recognition.detector import PlateDetector
from plate_recognition.recognizer import CharRecognizer

import sys, os, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plate_recognition.detector import PlateDetector
from plate_recognition.recognizer import CharRecognizer

img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
for fname in ["car.jpg", "car4.jpg"]:
    print(f"\n=== {fname} ===")
    img = cv2.imread(os.path.join(img_dir, fname))
    det = PlateDetector()
    cands = det.detect(img)
    r = CharRecognizer()
    r.load_models()
    for i, (b, p) in enumerate(cands[:4]):
        gray = r._to_gray(p)
        is_dark = gray.mean() < 80
        print(f"  [{i}] shape={p.shape}, dark={is_dark}")
        for j, proc in enumerate(r._get_strategies(p, is_dark)):
            res = r._reader.readtext(proc, detail=0)
            label = ["gray","binary","bright","gamma"][j] if j < 4 else f"s{j}"
            print(f"    {label}: {res}")
        fin = r.recognize_plate(p)
        print(f"    Final: {repr(fin)}")
