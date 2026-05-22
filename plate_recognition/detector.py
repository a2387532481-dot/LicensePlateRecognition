import cv2
import numpy as np


class PlateDetector:
    def __init__(self):
        # 极度宽泛的蓝色 HSV 范围，能兼容黑暗奥迪和高光粤B，但绝对能屏蔽树叶和白墙
        self.lower_blue = np.array([100, 40, 40])
        self.upper_blue = np.array([140, 255, 255])

    def detect(self, image: np.ndarray) -> list:
        # 1. 纹理初筛：用 Sobel 找出所有长得像车牌的候选框（包含车牌、进气格栅、树影）
        candidates = self._detect_by_sobel(image)

        valid_plates = []
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 2. 颜色复核：是李逵还是李鬼，拿颜色一验便知
        for bbox, plate_img in candidates:
            x, y, w, h = bbox

            # 只取候选框最核心的区域（四周往里缩进20%），避开边框反光和外部杂质
            cx_start, cx_end = int(w * 0.2), int(w * 0.8)
            cy_start, cy_end = int(h * 0.2), int(h * 0.8)

            if cx_end <= cx_start or cy_end <= cy_start:
                continue

            # 提取核心区域的 HSV 色彩
            core_hsv = cv2.cvtColor(plate_img[cy_start:cy_end, cx_start:cx_end], cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(core_hsv, self.lower_blue, self.upper_blue)

            # 计算核心区域的蓝色像素占比
            blue_ratio = cv2.countNonZero(blue_mask) / (core_hsv.shape[0] * core_hsv.shape[1] + 1e-5)

            # 如果核心区域连 15% 的蓝色都没有（比如树影、黑色进气格栅），直接淘汰！
            if blue_ratio > 0.15:
                valid_plates.append((bbox, plate_img, blue_ratio))

        # 按 蓝色纯度 * 面积 进行综合打分排序，确保最蓝最大的排第一
        valid_plates.sort(key=lambda item: item[2] * item[0][2] * item[0][3], reverse=True)

        kept = []
        for item in valid_plates:
            bbox, plate_img, _ = item
            if not self._overlaps_any(bbox, [k[0] for k in kept]):
                kept.append((bbox, plate_img))

        return kept[:3]

    def _detect_by_sobel(self, image: np.ndarray) -> list:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        sobel_x = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)
        abs_x = cv2.convertScaleAbs(sobel_x)
        _, binary = cv2.threshold(abs_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 横向膨胀把文字连成一片
        kernel_w = max(15, int(image.shape[1] * 0.02))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / max(h, 1)

            # 宽松的几何过滤
            if 1000 < w * h < 150000 and 1.8 < ratio < 5.5:
                plate_img = image[y:y + h, x:x + w]
                results.append(((x, y, w, h), plate_img))

        return results

    def _overlaps_any(self, bbox, existing: list, iou_thresh: float = 0.3) -> bool:
        if not existing: return False
        for eb in existing:
            if self._iou(bbox, eb) > iou_thresh: return True
        return False

    def _iou(self, a, b) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0: return 0.0
        return inter / max(min(a[2] * a[3], b[2] * b[3]), 1)