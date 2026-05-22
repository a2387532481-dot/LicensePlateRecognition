import cv2
import numpy as np
# 导入你的配置，看看它到底是多少
from plate_recognition.config import BLUE_LOWER1, BLUE_UPPER1

# 读入你这张测试图
img_path = r"LicensePlateRecognition\images\car.jpg"
img = cv2.imread(img_path)

if img is None:
    print("找不到图片，请检查路径")
else:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 用你现有的配置生成面具
    mask = cv2.inRange(hsv, BLUE_LOWER1, BLUE_UPPER1)

    # 保存出来看一眼
    cv2.imwrite("debug_mask.jpg", mask)
    print("--- 调试信息 ---")
    print("你当前的 BLUE_LOWER1:", BLUE_LOWER1)
    print("你当前的 BLUE_UPPER1:", BLUE_UPPER1)
    print("面具图片已生成，请去项目根目录打开 debug_mask.jpg 看看！")
