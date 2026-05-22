"""
基于OpenCV的车牌号识别系统

功能：检测图像中的车牌 → 分割字符 → 识别字符
支持中国蓝色标准车牌（省份简称 + 字母 + 5位字母数字）

使用方法:
  1. 训练模型:     python -m plate_recognition.train
  2. 识别车牌:     python 车牌识别系统.py <图片路径>
     或:          python -m plate_recognition.main <图片路径>

模块:
  plate_recognition/
  ├── config.py       — 配置参数
  ├── detector.py     — 车牌检测（颜色+边缘）
  ├── segmenter.py    — 字符分割
  ├── model.py        — CNN模型定义+训练工具
  ├── recognizer.py   — 字符识别
  ├── train.py        — 训练脚本
  └── main.py         — 主流程
"""

from plate_recognition.main import main

if __name__ == "__main__":
    main()
