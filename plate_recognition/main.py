"""
Main pipeline: detect → segment → recognize.
"""
import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plate_recognition.detector import PlateDetector
from plate_recognition.segmenter import CharSegmenter
from plate_recognition.recognizer import CharRecognizer


class LicensePlateSystem:
    """Complete license plate recognition pipeline."""

    def __init__(self, device: str = "cpu"):
        self.detector = PlateDetector()
        self.segmenter = CharSegmenter()
        self.recognizer = CharRecognizer(device=device)

    def initialize(self):
        """Load CNN recognition models."""
        self.recognizer.load_cnn_models()

    def recognize(self, image_path: str, visualize: bool = True) -> dict:
        """
        Run end-to-end recognition on an image.
        Returns a dict with keys: plate_text, candidates, plate_img.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        # Step 1: Detect plates
        candidates = self.detector.detect(image)

        if not candidates:
            print("No license plate detected.")
            return {"plate_text": "", "candidates": [], "plate_img": None}

        results = []
        for bbox, plate_img in candidates:
            # Step 2: Segment characters
            chars = self.segmenter.segment(plate_img)
            char_images = [c[0] for c in chars] if chars else []

            # Step 3: CNN-based per-character recognition
            plate_text = self.recognizer.recognize(char_images)

            # Step 4: Fallback to PaddleOCR if CNN fails
            if not plate_text or "?" in plate_text:
                paddle_text = self.recognizer.recognize_plate(plate_img)
                if paddle_text and len(paddle_text) >= 5:
                    plate_text = paddle_text

            if plate_text:
                results.append({
                    "plate_text": plate_text,
                    "bbox": bbox,
                    "plate_img": plate_img,
                    "char_imgs": char_images,
                    "char_count": len(chars),
                })

        # Pick best result (longest valid text)
        if results:
            best = max(results, key=lambda r: len(r["plate_text"]))
        else:
            best = {"plate_text": "", "candidates": results, "plate_img": None}

        if visualize and best["plate_img"] is not None:
            self._visualize(image, best)

        return best

    def _visualize(self, original: np.ndarray, result: dict):
        """Display detection and recognition results."""
        display = original.copy()
        x, y, w, h = result["bbox"]

        # Draw bounding box
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw plate text using PIL (supports Chinese characters)
        text = result.get("plate_text", "")
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        display_pil = Image.fromarray(display_rgb)
        draw = ImageDraw.Draw(display_pil)
        try:
            font = ImageFont.truetype("simhei.ttf", 24)
        except Exception:
            try:
                font = ImageFont.truetype(r"C:\\Windows\\Fonts\\simhei.ttf", 24)
            except Exception:
                font = ImageFont.load_default()
        draw.text((x, y - 28), text, fill=(0, 255, 0), font=font)
        display = cv2.cvtColor(np.array(display_pil), cv2.COLOR_RGB2BGR)

        # Show plate ROI
        plate_display = result.get("plate_img")
        if plate_display is not None:
            h_plate, w_plate = plate_display.shape[:2]
            scale = 300 / max(h_plate, 1)
            plate_resized = cv2.resize(plate_display, (int(w_plate * scale), int(h_plate * scale)))
            cv2.imshow("Detected Plate", plate_resized)

        # Show character segments if available
        char_imgs = result.get("char_imgs", [])
        if char_imgs:
            char_tile = self._tile_chars(char_imgs)
            cv2.imshow("Characters", char_tile)

        cv2.imshow("License Plate Recognition", display)
        print(f"Recognized plate: {text}")
        # Also show ascii-safe representation for Chinese chars
        if any(ord(c) > 127 for c in text):
            print(f"  (chars: {' '.join(repr(c)[1:-1] if ord(c)>127 else c for c in text)})")
        print("Press any key to close windows…")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _tile_chars(self, char_imgs: list) -> np.ndarray:
        """Create a horizontal tiled image of character segments."""
        tiles = []
        for img in char_imgs:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            tiles.append(cv2.resize(img, (32, 40)))

        if not tiles:
            return np.zeros((40, 32, 3), dtype=np.uint8)

        return np.hstack(tiles)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="License Plate Recognition System")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization windows")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    args = parser.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    system = LicensePlateSystem(device=device)
    system.initialize()
    result = system.recognize(args.image, visualize=not args.no_visualize)

    if result["plate_text"]:
        print(f"\nFinal result: {result['plate_text']}")
    else:
        print("\nRecognition failed — no valid plate found.")


if __name__ == "__main__":
    main()
