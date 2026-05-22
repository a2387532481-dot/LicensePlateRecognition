"""
Diagnose PaddleOCR performance on license plate images.
Shows raw OCR results and saves detected plate crops for inspection.
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["KMP_AFFINITY"] = "disabled"
os.environ["OMP_NUM_THREADS"] = "1"

import paddle
paddle.set_flags({"FLAGS_use_mkldnn": 0})
from paddleocr import PaddleOCR

from plate_recognition.detector import PlateDetector


def diagnose(image_path: str):
    print(f"\n{'='*60}")
    print(f"Diagnosing: {image_path}")
    print(f"{'='*60}")

    img = cv2.imread(image_path)
    if img is None:
        print("ERROR: Cannot read image")
        return

    print(f"Image size: {img.shape}")

    # Step 1: Detect plate
    detector = PlateDetector()
    candidates = detector.detect(img)
    print(f"\nPlate candidates found: {len(candidates)}")

    if not candidates:
        print("No plate detected!")
        return

    # Step 2: Initialize PaddleOCR
    ocr = PaddleOCR(lang="ch")

    for i, (bbox, plate_img) in enumerate(candidates):
        print(f"\n--- Candidate {i} ---")
        print(f"  BBox: {bbox}")
        print(f"  Plate shape: {plate_img.shape}")

        # Save plate crop
        out_path = f"debug_paddle_plate_{i}.jpg"
        cv2.imwrite(out_path, plate_img)
        print(f"  Saved: {out_path}")

        # Run PaddleOCR on plate
        h, w = plate_img.shape[:2]
        if h < 200:
            scale = max(1.0, 200.0 / h)
            plate_resized = cv2.resize(plate_img, (int(w * scale), int(h * scale)),
                                       interpolation=cv2.INTER_CUBIC)
        else:
            plate_resized = plate_img

        if len(plate_resized.shape) == 3:
            img_rgb = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(plate_resized, cv2.COLOR_GRAY2RGB)

        results = ocr.ocr(img_rgb)
        print(f"\n  Raw PaddleOCR results:")
        if results and results[0]:
            for line in results[0]:
                box, (text, confidence) = line
                print(f"    text='{text}' conf={confidence:.3f} box_center_x={(box[0][0]+box[2][0])/2:.0f}")
        else:
            print("    (no text detected)")

        # Also try running PaddleOCR on the FULL original image
        print(f"\n--- PaddleOCR on full image ---")
        full_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        full_results = ocr.ocr(full_rgb)
        if full_results and full_results[0]:
            for line in full_results[0]:
                box, (text, confidence) = line
                print(f"    text='{text}' conf={confidence:.3f}")
        else:
            print("    (no text detected)")

        # Try different preprocessing
        print(f"\n--- Preprocessing variants ---")
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img

        variants = {
            "original": plate_img,
            "clahe": None,
            "invert": None,
            "otsu_binary": None,
        }

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        variants["clahe"] = clahe.apply(gray)
        variants["invert"] = 255 - gray
        _, variants["otsu_binary"] = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        for name, vimg in variants.items():
            if vimg is None:
                continue
            if len(vimg.shape) == 2:
                vimg_rgb = cv2.cvtColor(vimg, cv2.COLOR_GRAY2RGB)
            else:
                vimg_rgb = cv2.cvtColor(vimg, cv2.COLOR_BGR2RGB)
            vresults = ocr.ocr(vimg_rgb)
            texts = []
            if vresults and vresults[0]:
                for line in vresults[0]:
                    box, (text, conf) = line
                    texts.append(f"'{text}'({conf:.2f})")
            print(f"    {name:15s}: {', '.join(texts) if texts else '(none)'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image")
    args = parser.parse_args()
    diagnose(args.image)
