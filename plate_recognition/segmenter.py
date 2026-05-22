"""
Character segmenter — uses vertical projection analysis to find character regions.
Adapted from the CCPD preprocessing approach for robust real-world segmentation.
"""
import cv2
import numpy as np
from .config import CHAR_WIDTH, CHAR_HEIGHT


class CharSegmenter:
    """Segment individual characters from a license plate region."""

    def __init__(self):
        pass

    def segment(self, plate_img: np.ndarray) -> list:
        """Extract 7 character images from a plate using vertical projection.

        Returns list of (char_normalized_28x28, x0_offset) tuples.
        """
        if plate_img is None or plate_img.size == 0:
            return []

        h, w = plate_img.shape[:2]
        if h < 10 or w < 30:
            return []

        # Trim borders
        trim_t = max(1, int(h * 0.08))
        trim_b = max(1, int(h * 0.08))
        trim_l = max(1, int(w * 0.02))
        trim_r = max(1, int(w * 0.02))
        plate_t = plate_img[trim_t:h - trim_b, trim_l:w - trim_r]
        h_t, w_t = plate_t.shape[:2]

        if h_t < 8 or w_t < 30:
            return []

        # Convert to grayscale — use full conversion, not just red channel
        if len(plate_t.shape) == 3:
            gray = cv2.cvtColor(plate_t, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_t

        # Try vertical projection to find character regions
        char_regions = self._find_chars_by_projection(gray, w_t)

        if char_regions and len(char_regions) == 7:
            # Match regions to ideal positions (left to right)
            ideal_x = np.linspace(0.05, 0.95, 7) * w_t
            result = []
            used = set()
            for pos in range(7):
                target_x = ideal_x[pos]
                best_idx = -1
                best_dist = float("inf")
                for i, (s, e) in enumerate(char_regions):
                    if i in used:
                        continue
                    dist = abs((s + e) / 2 - target_x)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                if best_idx >= 0:
                    used.add(best_idx)
                    cx1, cx2 = char_regions[best_idx]
                    # Extract with some padding
                    pad = int((cx2 - cx1) * 0.15)
                    x1 = max(0, cx1 - pad)
                    x2 = min(w_t, cx2 + pad)
                    char_img = gray[0:h_t, x1:x2]
                    char_norm = self._normalize(char_img)
                    if char_norm.mean() > 0.005:
                        result.append((char_norm, x1 + trim_l))
                    else:
                        result.append((np.zeros((CHAR_HEIGHT, CHAR_WIDTH), dtype=np.float32), x1 + trim_l))
        else:
            # Fallback: uniform slot division with dot separator awareness
            result = self._uniform_fallback(gray, trim_l)

        return result

    def _find_chars_by_projection(self, gray: np.ndarray, w: int) -> list:
        """Find character regions using vertical projection of inverted binary."""
        # OTSU binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Check if text is dark-on-light or light-on-dark
        # For blue plates, characters are typically white (light) on blue (darker)
        dark_pixels = (binary == 0).sum()
        light_pixels = (binary == 255).sum()
        if light_pixels > dark_pixels:
            # Text is dark on light background — invert
            binary = 255 - binary

        # Vertical projection: sum dark (text) pixels per column
        proj = (binary == 0).sum(axis=0).astype(np.float64)

        # Smooth the projection
        ks = max(3, w // 60)
        smoothed = np.convolve(proj, np.ones(ks) / ks, mode='same')

        # Find regions above threshold
        threshold = max(smoothed.max() * 0.12, 2.0)
        above = smoothed > threshold

        char_regions = []
        in_char = False
        start_x = 0
        for x in range(w):
            if above[x] and not in_char:
                start_x = x
                in_char = True
            elif not above[x] and in_char:
                if x - start_x >= w * 0.015:  # minimum char width
                    char_regions.append((start_x, x))
                in_char = False
        if in_char and w - start_x >= w * 0.015:
            char_regions.append((start_x, w))

        # If we found too many regions, try merging close ones
        if len(char_regions) > 7:
            char_regions = self._merge_close_regions(char_regions, w)

        return char_regions

    def _merge_close_regions(self, regions: list, w: int) -> list:
        """Merge character regions that are very close together (likely noise splits)."""
        if len(regions) <= 7:
            return regions

        merged = []
        i = 0
        while i < len(regions):
            if i < len(regions) - 1:
                gap = regions[i + 1][0] - regions[i][1]
                if gap < w * 0.02:  # very close — merge
                    merged.append((regions[i][0], regions[i + 1][1]))
                    i += 2
                else:
                    merged.append(regions[i])
                    i += 1
            else:
                merged.append(regions[i])
                i += 1

        # If still too many, keep the 7 widest
        if len(merged) > 7:
            merged.sort(key=lambda r: r[1] - r[0], reverse=True)
            merged = sorted(merged[:7])

        return merged

    def _uniform_fallback(self, gray: np.ndarray, trim_l: int = 0) -> list:
        """Fallback: uniform slot division with Chinese plate dot separator awareness.

        Chinese blue plate layout:
          - Positions 0-1: two chars before the dot (province + letter)
          - Dot separator at ~28.5% of plate width
          - Positions 2-6: five chars after the dot
        """
        h, w = gray.shape

        margin_left = int(w * 0.02)
        margin_right = int(w * 0.01)
        dot_x = int(w * 0.285)  # dot separator position
        dot_half = max(1, int(h * 0.12))  # half dot diameter

        left_w = dot_x - dot_half - margin_left
        right_w = w - dot_x - dot_half - margin_right
        v_trim_top = int(h * 0.12)
        v_trim_bottom = int(h * 0.12)

        # First 2 chars (positions 0-1) in left section
        boxes = []
        for i in range(2):
            cx = margin_left + int(left_w * (i + 0.5) / 2)
            hw = max(int(left_w / 2 * 0.45), 6)
            boxes.append((max(0, cx - hw), min(w, cx + hw)))

        # Last 5 chars (positions 2-6) in right section
        start_x = dot_x + dot_half
        for i in range(5):
            cx = start_x + int(right_w * (i + 0.5) / 5)
            hw = max(int(right_w / 5 * 0.45), 6)
            boxes.append((max(0, cx - hw), min(w, cx + hw)))

        result = []
        for pos, (x1, x2) in enumerate(boxes):
            char_img = gray[v_trim_top:h - v_trim_bottom, x1:x2]
            char_norm = self._normalize(char_img)
            result.append((char_norm, x1 + trim_l))

        return result

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize character image to match CCPD training preprocessing EXACTLY.

        Training pipeline: preprocess_char(target_size=(64,64)) → cv2.resize(28,28,INTER_AREA) → /255.0

        We reproduce the identical steps here so inference matches training.
        """
        if img is None or img.size == 0 or img.shape[0] < 3 or img.shape[1] < 3:
            return np.zeros((CHAR_HEIGHT, CHAR_WIDTH), dtype=np.float32)

        # Step 1: CLAHE enhancement (identical to preprocess_char)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(img)

        # Step 2: OTSU binarization (identical to preprocess_char)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 3: ensure white text on black background
        if (binary == 255).sum() > (binary == 0).sum():
            binary = 255 - binary

        # Step 4: tight crop with 15% padding (identical to preprocess_char)
        coords = cv2.findNonZero(binary)
        if coords is None:
            # Retry with lower threshold for thin strokes
            _, binary = cv2.threshold(enhanced, int(enhanced.mean() * 0.7), 255,
                                      cv2.THRESH_BINARY)
            if (binary == 255).sum() > (binary == 0).sum():
                binary = 255 - binary
            coords = cv2.findNonZero(binary)

        if coords is None:
            # Last resort: use enhanced grayscale, resize to 64 then 28
            enhanced_64 = cv2.resize(enhanced, (64, 64), interpolation=cv2.INTER_CUBIC)
            result = cv2.resize(enhanced_64, (CHAR_WIDTH, CHAR_HEIGHT), interpolation=cv2.INTER_AREA)
            return result.astype(np.float32) / 255.0

        x, y, bw, bh = cv2.boundingRect(coords)
        px, py = int(bw * 0.15), int(bh * 0.15)
        x1, y1 = max(0, x - px), max(0, y - py)
        x2, y2 = min(img.shape[1], x + bw + px), min(img.shape[0], y + bh + py)
        tight = binary[y1:y2, x1:x2]

        # Step 5: resize to 64x64 with margin=8 (identical to preprocess_char target_size=64)
        margin = 8
        size64 = 64
        avail = size64 - 2 * margin
        h_t, w_t = tight.shape[:2]
        scale = min(avail / h_t, avail / w_t)
        new_h, new_w = max(4, int(h_t * scale)), max(4, int(w_t * scale))
        resized_64 = np.zeros((size64, size64), dtype=np.uint8)
        resized = cv2.resize(tight, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        y_off = (size64 - new_h) // 2
        x_off = (size64 - new_w) // 2
        resized_64[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # Step 6: resize 64→28 with INTER_AREA (identical to training data loader)
        result = cv2.resize(resized_64, (CHAR_WIDTH, CHAR_HEIGHT), interpolation=cv2.INTER_AREA)

        # Step 7: normalize to [0, 1] (identical to training)
        return result.astype(np.float32) / 255.0

    # Legacy compatibility stubs
    def _make_binary(self, *args, **kwargs):
        return np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10), dtype=np.uint8)

    def _extract_by_contours(self, *args, **kwargs):
        return []
