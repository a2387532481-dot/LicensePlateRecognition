"""
License plate recognition system configuration.
"""
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "plate_recognition", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DIGIT_MODEL_PATH = os.path.join(MODEL_DIR, "digit_cnn.pth")
LETTER_MODEL_PATH = os.path.join(MODEL_DIR, "letter_cnn.pth")
PROVINCE_MODEL_PATH = os.path.join(MODEL_DIR, "province_cnn.pth")
ALPHANUM_MODEL_PATH = os.path.join(MODEL_DIR, "alphanum_cnn.pth")

# 极度放宽 HSV 限制，只要带点蓝色调全放行（拯救阴影里的京K）
BLUE_LOWER1 = np.array([100, 40, 40])
BLUE_UPPER1 = np.array([130, 255, 255])

BLUE_LOWER2 = np.array([100, 40, 40])
BLUE_UPPER2 = np.array([130, 255, 255])

PLATE_MIN_AREA = 1000
PLATE_MAX_AREA = 200000
PLATE_MIN_RATIO = 1.5
PLATE_MAX_RATIO = 5.0

CHAR_WIDTH = 28
CHAR_HEIGHT = 28
CHAR_MIN_AREA = 30
CHAR_MAX_AREA = 2000
CHAR_MIN_RATIO = 0.15
CHAR_MAX_RATIO = 1.2

PROVINCES = [
    "京", "津", "沪", "渝", "冀", "豫", "云", "辽", "黑", "湘",
    "皖", "鲁", "新", "苏", "浙", "赣", "鄂", "桂", "甘", "晋",
    "蒙", "陕", "吉", "闽", "贵", "粤", "青", "藏", "川", "宁", "琼"
]

LETTERS = [chr(ord("A") + i) for i in range(26) if chr(ord("A") + i) not in ("I", "O")]
DIGITS = [str(i) for i in range(10)]
ALPHANUMERIC = DIGITS + LETTERS
