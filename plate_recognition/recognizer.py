import numpy as np
from paddleocr import PaddleOCR


class CharRecognizer:
    def __init__(self):
        self.ocr = PaddleOCR(lang="ch", enable_mkldnn=False)
        self._loaded = True

    def load_models(self):
        pass

    def recognize_plate(self, plate_img: np.ndarray) -> str:
        if plate_img is None or plate_img.size == 0:
            return ""

        try:
            result = self.ocr.predict(plate_img)
            if result and result[0]['rec_texts']:
                text = result[0]['rec_texts'][0]
                return self._apply_plate_rules(self._clean(text))
        except Exception as e:
            print(f"PaddleOCR 出错了: {e}")

        return "识别失败"

    def recognize(self, char_images: list) -> str:
        return ""

    def _clean(self, text: str) -> str:
        text = text.strip().upper()
        for ch in " .-·•,:/_|\\'\"[]()@#$%^&*+=<>{}~`!?;":
            text = text.replace(ch, "")
        return text

    def _apply_plate_rules(self, text: str) -> str:
        if len(text) < 2:
            return text
        chars = list(text)
        # 强行纠正：车牌第二位必须是字母
        digit_to_letter = {"0": "D", "1": "L", "2": "Z", "5": "S", "8": "B"}
        if chars[1] in "0123456789":
            chars[1] = digit_to_letter.get(chars[1], chars[1])
        return "".join(chars)