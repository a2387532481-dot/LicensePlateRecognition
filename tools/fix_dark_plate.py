"""Fix for dark license plate images."""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plate_recognition.detector import PlateDetector
from plate_recognition.recognizer import CharRecognizer

img = cv2.imread(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images", "car3.jpg"))
det = PlateDetector()
cands = det.detect(img)
_, plate = cands[1]  # candidate[1] was the one with text

gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# Upscale significantly
scale = 300 / h
gray_up = cv2.resize(gray, (int(w * scale), 300), interpolation=cv2.INTER_CUBIC)

# Try ALL preprocessing methods
methods = {}

# 1. Strong brightening
bright = cv2.convertScaleAbs(gray_up, alpha=2.8, beta=50)
methods["strong bright"] = bright

# 2. Auto gamma
mean_val = gray_up.mean()
gamma = np.log(128) / np.log(mean_val) if mean_val > 0 else 1.0
gamma = np.clip(gamma, 0.3, 3.0)
lut = np.array([((i / 255) ** (1 / gamma)) * 255 for i in range(256)], dtype="uint8")
gamma_corrected = cv2.LUT(gray_up, lut)
methods["gamma"] = gamma_corrected

# 3. CLAHE strong
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
methods["CLAHE strong"] = clahe.apply(gray_up)

# 4. Histogram equalization
methods["hist eq"] = cv2.equalizeHist(gray_up)

# 5. Adaptive threshold (binary inverted)
th = cv2.adaptiveThreshold(gray_up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv2.THRESH_BINARY, 15, 3)
methods["adapt thresh"] = th
methods["adapt inv"] = 255 - th

# 6. Otsu
_, otsu = cv2.threshold(gray_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
methods["otsu"] = otsu
methods["otsu inv"] = 255 - otsu

r = CharRecognizer()
r.load_models()

print(f"Plate: {plate.shape}, upscaled to {gray_up.shape}")
print(f"Gray mean: {gray_up.mean():.0f}, min={gray_up.min()}, max={gray_up.max()}")
print()

results = {}
for name, proc in methods.items():
    if len(proc.shape) == 2:
        proc_rgb = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)
    else:
        proc_rgb = proc
    res = r._reader.readtext(proc_rgb, detail=0)
    text = "".join(res) if res else "(empty)"
    results[name] = text
    print(f"  {name:15s}: {text}")

# Also try on the best method with allowlist
best_methods = [k for k, v in results.items() if v and v != "(empty)"]
print("\nBest results:")
for m in best_methods[:3]:
    print(f"  {m}: {results[m]}")
