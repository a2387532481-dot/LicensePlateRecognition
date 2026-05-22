"""
CCPD dataset preprocessing for Chinese blue license plate recognition training.

Targets CCPD-Base subset (standard plates for train/val). Other subsets
(db, blur, fn, rotate, tilt, challenge) are used as test data.

CCPD default directory structure:
    CCPD2019/
    ├── ccpd_base/     ←  training data (~200k images)
    ├── ccpd_db/       ←  test: uneven lighting
    ├── ccpd_blur/     ←  test: motion blur
    ├── ccpd_fn/       ←  test: far/near distance
    ├── ccpd_rotate/   ←  test: rotated plates
    ├── ccpd_tilt/     ←  test: tilted plates
    └── ccpd_challenge/←  test: hardest cases

Usage:
    # Train set only: 5000 base plates
    python tools/preprocess_ccpd.py --ccpd_dir D:/CCPD2019 --out_dir D:/ccpd_train

    # Train + val split: 8000 base plates, 80/20 split
    python tools/preprocess_ccpd.py --ccpd_dir D:/CCPD2019 --out_dir D:/ccpd_train --max 8000 --val_split 0.2

    # With individual characters extracted
    python tools/preprocess_ccpd.py --ccpd_dir D:/CCPD2019 --out_dir D:/ccpd_train --max 5000 --chars_only

    # Extract test subsets (small sample from each)
    python tools/preprocess_ccpd.py --ccpd_dir D:/CCPD2019 --out_dir D:/ccpd_test --subset db,blur,fn --max 500
"""
import argparse
import csv
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# ── CCPD subset directories ───────────────────────────────────────────
CCPD_SUBSETS = {
    "base": "ccpd_base",
    "db": "ccpd_db",
    "blur": "ccpd_blur",
    "fn": "ccpd_fn",
    "rotate": "ccpd_rotate",
    "tilt": "ccpd_tilt",
    "challenge": "ccpd_challenge",
    "green": "ccpd_green",  # new energy vehicles, excluded by default
}

# ── CCPD character tables ─────────────────────────────────────────────
PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
]  # 0-30 valid, 31 = "O" placeholder

ALPHABETS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z',
]  # 0-23 valid, 24 = 'O' placeholder

ADS = [  # letters + digits for positions 2-6
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
]  # 0-33 valid, 34 = 'O' placeholder

# ── Filename parsing ──────────────────────────────────────────────────


def parse_ccpd_filename(filename: str) -> dict:
    """Parse CCPD filename into structured annotation.

    Example: 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg

    Fields: area-tilt-bbox-vertices-lp_char_indices-brightness-blurriness
    Vertices order: right-bottom, left-bottom, left-up, right-up
    """
    name = Path(filename).stem
    parts = name.split("-")
    if len(parts) < 7:
        return None

    try:
        area = float(parts[0])
        tilt_h, tilt_v = map(float, parts[1].split("_"))

        bbox_lu, bbox_rb = parts[2].split("_")
        bbox = [
            tuple(map(int, bbox_lu.split("&"))),
            tuple(map(int, bbox_rb.split("&"))),
        ]

        vertices_raw = parts[3].split("_")
        vertices = []
        for v in vertices_raw:
            x, y = v.split("&")
            vertices.append((int(x), int(y)))

        char_indices = list(map(int, parts[4].split("_")))
        brightness = int(parts[5])
        blurriness = int(parts[6])

        plate_number = _decode_plate(char_indices)
        if plate_number is None:
            return None

        return {
            "filename": filename,
            "area_ratio": area,
            "tilt_h": tilt_h,
            "tilt_v": tilt_v,
            "bbox": bbox,
            "vertices": vertices,
            "char_indices": char_indices,
            "plate_number": plate_number,
            "brightness": brightness,
            "blurriness": blurriness,
        }
    except (ValueError, IndexError):
        return None


def _decode_plate(indices: list) -> str:
    """Decode plate number from CCPD character indices."""
    if len(indices) < 7:
        return None
    chars = []
    # Position 0: province
    if indices[0] < len(PROVINCES):
        chars.append(PROVINCES[indices[0]])
    else:
        return None
    # Position 1: letter
    if indices[1] < len(ALPHABETS):
        chars.append(ALPHABETS[indices[1]])
    else:
        return None
    # Positions 2-6: alphanumeric
    for i in range(2, 7):
        idx = indices[i]
        if idx >= len(ADS) or ADS[idx] == 'O':
            return None
        chars.append(ADS[idx])

    result = "".join(chars)
    if len(result) != 7:
        return None
    return result

# ── Filters ───────────────────────────────────────────────────────────


def is_blue_plate(ann: dict) -> bool:
    """Standard blue plate: 7 chars, province + letter + 5 alphanumeric."""
    plate = ann["plate_number"]
    return (plate is not None
            and len(plate) == 7
            and plate[0] in PROVINCES
            and plate[1] in ALPHABETS)


def quality_ok(ann: dict) -> bool:
    """Reject bad samples: too blurry, extreme tilt, too dark/bright.

    CCPD tilt values are camera angles where 90° = perpendicular to plate.
    We reject plates tilted more than 30° from frontal.
    """
    if ann["blurriness"] > 80:
        return False
    if abs(ann["tilt_h"] - 90) > 30 or abs(ann["tilt_v"] - 90) > 30:
        return False
    if ann["brightness"] < 10 or ann["brightness"] > 200:
        return False
    return True

# ── Perspective crop ──────────────────────────────────────────────────


def perspective_crop(img: np.ndarray, vertices: list,
                     output_size=(440, 140)) -> np.ndarray:
    """Four-point perspective correction → straight plate image.

    CCPD vertices order: rb, lb, lu, ru → mapped to output corners.
    """
    src_pts = np.float32(vertices)
    out_w, out_h = output_size
    dst_pts = np.float32([
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
        [0, 0],
        [out_w - 1, 0],
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, output_size, borderMode=cv2.BORDER_REPLICATE)

# ── Character segmentation ────────────────────────────────────────────


def segment_characters(plate_img: np.ndarray, plate_number: str) -> list:
    """Segment 7 characters from a straightened plate image.

    Uses vertical projection of inverted binary to find character regions,
    then assigns each to the closest ideal position (Chinese plate layout).
    Returns list of (position, char_label, char_gray_image).
    """
    if len(plate_number) != 7:
        return []

    gray = (cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            if len(plate_img.shape) == 3 else plate_img)
    h, w = gray.shape

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    proj = (binary == 0).sum(axis=0).astype(np.float64)

    ks = max(3, w // 60)
    smoothed = np.convolve(proj, np.ones(ks) / ks, mode='same')

    threshold = smoothed.max() * 0.15
    above = smoothed > threshold

    char_regions = []
    in_char = False
    start_x = 0
    for x in range(w):
        if above[x] and not in_char:
            start_x = x
            in_char = True
        elif not above[x] and in_char:
            if x - start_x >= w * 0.02:
                char_regions.append((start_x, x))
            in_char = False
    if in_char and w - start_x >= w * 0.02:
        char_regions.append((start_x, w))

    if len(char_regions) != 7:
        return _uniform_segment(gray, plate_number)

    chars = list(plate_number)
    ideal_x = np.linspace(0.05, 0.95, 7) * w
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
            char_img = gray[:, cx1:cx2]
            result.append((pos, chars[pos], char_img))

    return result


def _uniform_segment(gray: np.ndarray, plate_number: str) -> list:
    """Fallback: uniform slot division for Chinese blue plate layout."""
    h, w = gray.shape
    chars = list(plate_number)

    margin_left = int(w * 0.02)
    margin_right = int(w * 0.01)
    dot_x = int(w * 0.285)
    dot_half = int(h * 0.12)

    left_w = dot_x - dot_half - margin_left
    right_w = w - dot_x - dot_half - margin_right
    v_trim_top = int(h * 0.12)
    v_trim_bottom = int(h * 0.12)

    boxes = []
    for i in range(2):
        cx = margin_left + int(left_w * (i + 0.5) / 2)
        hw = max(int(left_w / 2 * 0.44), 8)
        boxes.append((max(0, cx - hw), min(w, cx + hw)))
    start_x = dot_x + dot_half
    for i in range(5):
        cx = start_x + int(right_w * (i + 0.5) / 5)
        hw = max(int(right_w / 5 * 0.44), 8)
        boxes.append((max(0, cx - hw), min(w, cx + hw)))

    result = []
    for pos, (x1, x2) in enumerate(boxes):
        char_img = gray[v_trim_top:h - v_trim_bottom, x1:x2]
        result.append((pos, chars[pos], char_img))
    return result


def preprocess_char(char_img: np.ndarray, target_size=(64, 64)) -> np.ndarray:
    """Normalize single char: CLAHE → binarize (invert if needed) → tight crop → resize."""
    if char_img is None or char_img.size == 0 or char_img.shape[0] < 4 or char_img.shape[1] < 4:
        return np.zeros(target_size, dtype=np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(char_img)

    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure white text on black background
    if (binary == 255).sum() > (binary == 0).sum():
        binary = 255 - binary

    coords = cv2.findNonZero(binary)
    if coords is None:
        # Chinese province chars may have thin strokes lost in Otsu;
        # retry with a lower fixed threshold to capture fine strokes
        _, binary = cv2.threshold(enhanced, int(enhanced.mean() * 0.7), 255,
                                  cv2.THRESH_BINARY)
        if (binary == 255).sum() > (binary == 0).sum():
            binary = 255 - binary
        coords = cv2.findNonZero(binary)
    if coords is None:
        # Last resort: use the CLAHE-enhanced grayscale directly
        gray_norm = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        margin = 4
        h, w = gray_norm.shape
        scale = min((target_size[0] - 2 * margin) / h,
                    (target_size[1] - 2 * margin) / w)
        new_h, new_w = max(4, int(h * scale)), max(4, int(w * scale))
        resized = cv2.resize(gray_norm, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        canvas = np.zeros(target_size, dtype=np.uint8)
        y_off = (target_size[0] - new_h) // 2
        x_off = (target_size[1] - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    x, y, bw, bh = cv2.boundingRect(coords)
    px, py = int(bw * 0.15), int(bh * 0.15)
    x1, y1 = max(0, x - px), max(0, y - py)
    x2, y2 = min(char_img.shape[1], x + bw + px), min(char_img.shape[0], y + bh + py)
    tight = binary[y1:y2, x1:x2]

    margin = 8
    avail = target_size[0] - 2 * margin
    scale = min(avail / tight.shape[0], avail / tight.shape[1])
    new_h, new_w = max(4, int(tight.shape[0] * scale)), max(4, int(tight.shape[1] * scale))
    resized = cv2.resize(tight, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.zeros(target_size, dtype=np.uint8)
    y_off = (target_size[0] - new_h) // 2
    x_off = (target_size[1] - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return canvas

# ── Main pipeline ─────────────────────────────────────────────────────


def collect_images(ccpd_dir: str, subsets: list) -> list:
    """Collect image paths from specified CCPD subset directories."""
    ccpd_root = Path(ccpd_dir)
    image_files = []

    for subset in subsets:
        subdir_name = CCPD_SUBSETS.get(subset, subset)
        subdir = ccpd_root / subdir_name
        if subdir.is_dir():
            imgs = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
            image_files.extend(imgs)
            print(f"  {subset}: {len(imgs)} images from {subdir_name}/")
        else:
            print(f"  {subset}: NOT FOUND ({subdir_name}/ missing)")

    return image_files


def process_dataset(ccpd_dir: str, out_dir: str, max_samples: int,
                    chars_only: bool = False, subsets: list = None,
                    val_split: float = 0.0):
    """Process CCPD images, extracting blue plate crops and (optionally) characters.

    Args:
        ccpd_dir: root CCPD directory
        out_dir: output directory
        max_samples: max blue plates to save
        chars_only: also extract individual character crops
        subsets: list of subset keys (default: ["base"])
        val_split: fraction for validation set (0 = train only)
    """
    if subsets is None:
        subsets = ["base"]

    print(f"Collecting images from CCPD subsets: {', '.join(subsets)}")
    image_files = collect_images(ccpd_dir, subsets)
    if not image_files:
        print("No images found.")
        return

    random.seed(42)
    random.shuffle(image_files)

    print(f"Total: {len(image_files)} images, targeting up to {max_samples} blue plates\n")

    out_path = Path(out_dir)

    # Decide output structure
    if val_split > 0:
        train_dir = out_path / "train"
        val_dir = out_path / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
    else:
        train_dir = out_path
        val_dir = None

    plates_dir = train_dir / "plates"
    plates_dir.mkdir(parents=True, exist_ok=True)

    labels_path = train_dir / "labels.csv"
    csv_f = open(labels_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    writer.writerow(["filename", "plate_number", "plate_file", "subset", "blurriness", "brightness"])

    val_csv = None
    val_writer = None
    if val_dir:
        val_plates_dir = val_dir / "plates"
        val_plates_dir.mkdir(parents=True, exist_ok=True)
        val_csv = open(val_dir / "labels.csv", "w", newline="", encoding="utf-8")
        val_writer = csv.writer(val_csv)
        val_writer.writerow(["filename", "plate_number", "plate_file", "subset", "blurriness", "brightness"])

    if chars_only:
        chars_dir = train_dir / "chars"
        chars_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "blue": 0, "saved": 0, "val_saved": 0, "segmented": 0,
             "skip_parse": 0, "skip_quality": 0, "skip_not_blue": 0}

    for img_path in image_files:
        if stats["saved"] + stats["val_saved"] >= max_samples:
            break

        stats["total"] += 1
        if stats["total"] % 1000 == 0:
            print(f"  Scanned {stats['total']} | blue={stats['blue']} | "
                  f"train={stats['saved']} val={stats['val_saved']}")

        ann = parse_ccpd_filename(img_path.name)
        if ann is None:
            stats["skip_parse"] += 1
            continue

        if not quality_ok(ann):
            stats["skip_quality"] += 1
            continue

        if not is_blue_plate(ann):
            stats["skip_not_blue"] += 1
            continue

        stats["blue"] += 1

        # Determine subset label from parent directory name
        subset_label = img_path.parent.name

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        try:
            plate_crop = perspective_crop(img, ann["vertices"])
        except Exception:
            continue

        # Train / val split decision
        is_val = val_split > 0 and random.random() < val_split

        if is_val:
            stats["val_saved"] += 1
            save_dir = val_plates_dir
            writer_use = val_writer
            plate_id = f"val_{stats['val_saved']:06d}"
        else:
            stats["saved"] += 1
            save_dir = plates_dir
            writer_use = writer
            plate_id = f"train_{stats['saved']:06d}"

        plate_filename = f"{plate_id}_{ann['plate_number']}.jpg"
        cv2.imwrite(str(save_dir / plate_filename), plate_crop)

        writer_use.writerow([
            img_path.name, ann["plate_number"], plate_filename, subset_label,
            ann["blurriness"], ann["brightness"],
        ])

        # Individual character extraction (train set only for cleaner data)
        if chars_only and not is_val:
            for pos, char, char_img in segment_characters(plate_crop, ann["plate_number"]):
                if char_img is None:
                    continue
                processed = preprocess_char(char_img)
                char_subdir = chars_dir / char
                char_subdir.mkdir(parents=True, exist_ok=True)
                out_path = char_subdir / f"{plate_id}_pos{pos}.png"
                success, buf = cv2.imencode('.png', processed)
                if success:
                    out_path.write_bytes(buf.tobytes())
            stats["segmented"] += 1

    csv_f.close()
    if val_csv:
        val_csv.close()

    print("\n" + "=" * 50)
    print("  CCPD Preprocessing Done")
    print("=" * 50)
    print(f"  Scanned:         {stats['total']}")
    print(f"  Parse errors:    {stats['skip_parse']}")
    print(f"  Quality reject:  {stats['skip_quality']}")
    print(f"  Non-blue skip:   {stats['skip_not_blue']}")
    print(f"  Blue plates:     {stats['blue']}")
    print(f"  Train saved:     {stats['saved']}")
    if val_dir:
        print(f"  Val saved:       {stats['val_saved']}")
    if chars_only:
        total_chars = sum(1 for _ in chars_dir.rglob("*.png"))
        print(f"  Chars saved:     {total_chars}")
    print(f"\n  Output → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CCPD preprocessing — extract blue plates for training"
    )
    parser.add_argument("--ccpd_dir", required=True,
                        help="CCPD dataset root directory (contains ccpd_base/ etc.)")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory")
    parser.add_argument("--subset", type=str, default="base",
                        help="CCPD subset(s) to use: base,db,blur,fn,rotate,tilt,challenge "
                             "(comma-separated, default: base)")
    parser.add_argument("--max", type=int, default=5000,
                        help="Max blue plates to save (default: 5000)")
    parser.add_argument("--val_split", type=float, default=0.0,
                        help="Validation split ratio (0-1, default: 0 = train only)")
    parser.add_argument("--chars_only", action="store_true",
                        help="Also extract individual character crops (train only)")
    args = parser.parse_args()

    if not os.path.isdir(args.ccpd_dir):
        print(f"Error: {args.ccpd_dir} not found")
        sys.exit(1)

    subsets = [s.strip() for s in args.subset.split(",")]
    process_dataset(args.ccpd_dir, args.out_dir, args.max,
                    args.chars_only, subsets, args.val_split)


if __name__ == "__main__":
    main()
