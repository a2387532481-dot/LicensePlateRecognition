import numpy as np
import torch
from paddleocr import PaddleOCR
from .model import CharCNN
from .config import (
    PROVINCE_MODEL_PATH, LETTER_MODEL_PATH, ALPHANUM_MODEL_PATH,
    PROVINCES, LETTERS, DIGITS,
)


class CharRecognizer:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.ocr = PaddleOCR(lang="ch")
        self.province_model = None
        self.letter_model = None
        self.alphanum_model = None
        self._cnn_loaded = False

    def load_models(self):
        self.province_model = CharCNN(len(PROVINCES)).to(self.device)
        self.province_model.load_state_dict(
            torch.load(PROVINCE_MODEL_PATH, map_location=self.device, weights_only=True))
        self.province_model.eval()

        self.letter_model = CharCNN(len(LETTERS)).to(self.device)
        self.letter_model.load_state_dict(
            torch.load(LETTER_MODEL_PATH, map_location=self.device, weights_only=True))
        self.letter_model.eval()

        alphanum_classes = DIGITS + LETTERS
        self.alphanum_model = CharCNN(len(alphanum_classes)).to(self.device)
        self.alphanum_model.load_state_dict(
            torch.load(ALPHANUM_MODEL_PATH, map_location=self.device, weights_only=True))
        self.alphanum_model.eval()

        self._cnn_loaded = True

    def recognize(self, char_images: list):
        """CNN per-character recognition.
        position 0 → province (31), 1 → letter (24), 2-6 → alphanum (34).
        Returns (plate_text, confidences_list).
        """
        if not self._cnn_loaded or len(char_images) < 7:
            return "", []

        # Only use first 7 chars
        images = char_images[:7]
        alphanum_classes = DIGITS + LETTERS
        results = []
        confs = []

        for i, img in enumerate(images):
            if img is None or img.size == 0 or img.mean() < 0.005:
                # Empty slot → use fallback
                results.append("?")
                confs.append(0.0)
                continue

            # Prepare tensor (1, 1, 28, 28)
            if img.ndim == 2:
                t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
            else:
                t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
                if t.shape[1] == 3:
                    t = t.mean(dim=1, keepdim=True)
            t = t.to(self.device)

            if i == 0:
                model = self.province_model
                classes = PROVINCES
            elif i == 1:
                model = self.letter_model
                classes = LETTERS
            else:
                model = self.alphanum_model
                classes = alphanum_classes

            with torch.no_grad():
                output = model(t)
                prob = torch.softmax(output, dim=1)
                conf, pred = torch.max(prob, dim=1)
                results.append(classes[pred.item()])
                confs.append(conf.item())

        text = "".join(results)
        return text, confs

    def recognize_plate(self, plate_img: np.ndarray, char_images: list = None):
        """Dual-engine: CNN first, PaddleOCR fallback.
        Returns dict with keys: plate_text, engine, confidences (optional).
        """
        # ── Try CNN ──
        if char_images and self._cnn_loaded:
            cnn_text, confs = self.recognize(char_images)
            real_count = sum(1 for c in cnn_text if c != "?")
            avg_conf = sum(confs) / max(len(confs), 1)
            if real_count >= 7 and avg_conf > 0.5:
                return {"plate_text": cnn_text, "engine": "CNN", "confidences": confs}

        # ── PaddleOCR fallback ──
        if plate_img is not None and plate_img.size > 0:
            try:
                result = self.ocr.predict(plate_img)
                if result and result[0].get("rec_texts"):
                    text = result[0]["rec_texts"][0]
                    text = self._apply_plate_rules(self._clean(text))
                    if text:
                        return {"plate_text": text, "engine": "PaddleOCR"}
            except Exception as e:
                print(f"PaddleOCR error: {e}")

        return {"plate_text": "识别失败", "engine": "PaddleOCR"}

    def _clean(self, text: str) -> str:
        text = text.strip().upper()
        for ch in " .-·•,:/_|\\'\"[]()@#$%^&*+=<>{}~`!?;":
            text = text.replace(ch, "")
        return text

    def _apply_plate_rules(self, text: str) -> str:
        if len(text) < 2:
            return text
        chars = list(text)
        digit_to_letter = {"0": "D", "1": "L", "2": "Z", "5": "S", "8": "B"}
        if chars[1] in "0123456789":
            chars[1] = digit_to_letter.get(chars[1], chars[1])
        return "".join(chars)
