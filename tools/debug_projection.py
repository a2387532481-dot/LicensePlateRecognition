"""Debug: vertical projection to find character positions."""
import cv2, sys, numpy as np
sys.path.insert(0, '.')
from plate_recognition.detector import PlateDetector
from plate_recognition.recognizer import CharRecognizer

detector = PlateDetector()
recognizer = CharRecognizer()
recognizer.load_models()

for car_name in ['car.jpg', 'car2.jpg']:
    img = cv2.imread(car_name)
    candidates = detector.detect(img)
    for bbox, plate_img in candidates[:1]:
        base = recognizer.preprocess(plate_img)
        h, w = base.shape
        print(f'{car_name}: {h}x{w}, mean={base.mean():.1f}')

        # Vertical projection: mean brightness per column
        v_proj = base.mean(axis=0)

        # Find peaks (bright = character, dark = background)
        # Smooth the projection
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(v_proj.astype(float), sigma=3.0)

        # Find local maxima (character centers)
        from scipy.signal import find_peaks
        threshold = smoothed.mean() + 0.3 * smoothed.std()
        peaks, props = find_peaks(smoothed, height=threshold, distance=w//12)

        print(f'  Found {len(peaks)} peaks at columns: {peaks.tolist()}')

        # Show projection as simplified ASCII chart
        norm = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-6)
        chart = ''
        for i in range(0, w, 8):
            val = norm[i]
            if val > 0.7: chart += '#'
            elif val > 0.5: chart += '='
            elif val > 0.3: chart += '-'
            else: chart += ' '
        print(f'  Projection: [{chart}]')
        print(f'  Dot at 28.5%: x={int(w*0.285)}')

        # Try a different approach: binary threshold and find contours
        _, thresh = cv2.threshold(base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Find connected components
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        # Filter by size (characters should be reasonable size)
        valid = []
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cx = centroids[i][0]
            if area > h * 5 and area < h * w * 0.3:  # reasonable char size
                valid.append((cx, area))
        valid.sort()
        print(f'  Connected components: {len(valid)} valid chars at x={[int(v[0]) for v in valid]}')
        print()
