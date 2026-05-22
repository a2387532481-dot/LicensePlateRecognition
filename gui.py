"""
License Plate Recognition GUI — load image, detect, recognize, visualize.
"""
import os
import sys

# Disable oneDNN/MKLDNN to avoid PaddlePaddle runtime errors
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["KMP_AFFINITY"] = "disabled"
os.environ["OMP_NUM_THREADS"] = "1"

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plate_recognition.detector import PlateDetector
from plate_recognition.segmenter import CharSegmenter
from plate_recognition.recognizer import CharRecognizer


class PlateRecognitionApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("车牌识别系统 — License Plate Recognition")
        self.root.geometry("1200x750")
        self.root.configure(bg="#f0f0f0")

        self.detector = PlateDetector()
        self.segmenter = CharSegmenter()
        self.recognizer = CharRecognizer()

        self.current_image = None
        self.result = None

        self._build_ui()

    def _build_ui(self):
        # ── Top toolbar ──
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(toolbar, text="打开图片", command=self._open_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="开始识别", command=self._recognize).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="清除", command=self._clear).pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(toolbar, text="就绪 — 请打开图片", foreground="gray")
        self.status_label.pack(side=tk.LEFT, padx=20)

        # ── Main area: image + results side by side ──
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left: image display
        left_frame = ttk.LabelFrame(main, text="原始图片 / 检测结果")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.image_canvas = tk.Canvas(left_frame, bg="#ffffff", highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right panel
        right_frame = ttk.Frame(main, width=320)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)

        # Result display
        result_frame = ttk.LabelFrame(right_frame, text="识别结果")
        result_frame.pack(fill=tk.X, padx=5, pady=5)

        self.result_var = tk.StringVar(value="—")
        result_label = tk.Label(
            result_frame, textvariable=self.result_var,
            font=("Microsoft YaHei", 28, "bold"),
            fg="#1a6e1a", bg="#ffffff",
            relief=tk.SUNKEN, anchor=tk.CENTER, padx=20, pady=15,
        )
        result_label.pack(fill=tk.X, padx=10, pady=10)

        # Detail info
        detail_frame = ttk.LabelFrame(right_frame, text="详细信息")
        detail_frame.pack(fill=tk.X, padx=5, pady=5)

        self.detail_text = tk.Text(detail_frame, height=6, width=35,
                                   font=("Consolas", 10), state=tk.DISABLED,
                                   bg="#fafafa", relief=tk.FLAT)
        self.detail_text.pack(fill=tk.BOTH, padx=5, pady=5)

        # Character display area
        char_frame = ttk.LabelFrame(right_frame, text="字符分割")
        char_frame.pack(fill=tk.X, padx=5, pady=5)

        self.char_canvas = tk.Canvas(char_frame, height=70, bg="#ffffff", highlightthickness=0)
        self.char_canvas.pack(fill=tk.X, padx=5, pady=5)

        # Plate ROI display
        plate_frame = ttk.LabelFrame(right_frame, text="车牌区域")
        plate_frame.pack(fill=tk.X, padx=5, pady=5)

        self.plate_canvas = tk.Canvas(plate_frame, height=60, bg="#ffffff", highlightthickness=0)
        self.plate_canvas.pack(fill=tk.X, padx=5, pady=5)

        # Status bar
        self.progress = ttk.Progressbar(self.root, mode="indeterminate")

    # ── Actions ──────────────────────────────────────────────────────

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.status_label.config(text="加载中…")
        self.root.update()

        self.current_image = cv2.imread(path)
        if self.current_image is None:
            messagebox.showerror("错误", f"无法读取图片: {path}")
            return

        self.result = None
        self._show_image(self.current_image)
        self.status_label.config(text=f"已加载: {os.path.basename(path)} — 点击「开始识别」")

    def _recognize(self):
        if self.current_image is None:
            messagebox.showwarning("提示", "请先打开一张图片")
            return

        self.progress.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=2)
        self.progress.start()
        self.status_label.config(text="识别中…")
        self.root.update()

        try:
            candidates = self.detector.detect(self.current_image)
            if not candidates:
                self.result_var.set("未检测到车牌")
                self.detail_text.config(state=tk.NORMAL)
                self.detail_text.delete(1.0, tk.END)
                self.detail_text.insert(tk.END, "未找到车牌区域")
                self.detail_text.config(state=tk.DISABLED)
                return

            bbox, plate_img = candidates[0]

            # Segment for visualization only
            chars = self.segmenter.segment(plate_img)
            char_images = [c[0] for c in chars] if chars else []

            # PaddleOCR recognition
            plate_text = self.recognizer.recognize_plate(plate_img)

            self.result = {
                "plate_text": plate_text,
                "bbox": bbox,
                "plate_img": plate_img,
                "char_imgs": char_images,
                "char_count": len(chars),
            }

            self._show_result()

        finally:
            self.progress.stop()
            self.progress.pack_forget()

    def _show_result(self):
        if not self.result:
            return

        text = self.result["plate_text"]
        self.result_var.set(text if text else "识别失败")
        self.status_label.config(text=f"识别完成: {text}")

        # Detail info
        details = []
        details.append(f"车牌号: {text}")
        details.append(f"字符数: {self.result['char_count']}")
        bbox = self.result["bbox"]
        details.append(f"位置: ({bbox[0]}, {bbox[1]}) {bbox[2]}x{bbox[3]}")
        details.append(f"候选数: 1")

        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)
        self.detail_text.insert(tk.END, "\n".join(details))
        self.detail_text.config(state=tk.DISABLED)

        # Show plate with bounding box
        self._show_detection()

        # Show character crops
        self._show_characters()

        # Show plate ROI
        self._show_plate_roi()

    def _show_detection(self):
        """Show original image with detection box and plate text overlay."""
        display = self.current_image.copy()
        x, y, w, h = self.result["bbox"]

        # Green bounding box
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Plate text above the box using PIL
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        display_pil = Image.fromarray(display_rgb)
        draw = ImageDraw.Draw(display_pil)
        try:
            font = ImageFont.truetype("simhei.ttf", 22)
        except Exception:
            try:
                font = ImageFont.truetype(r"C:\\Windows\\Fonts\\simhei.ttf", 22)
            except Exception:
                font = ImageFont.load_default()
        draw.text((x, y - 26), self.result["plate_text"], fill=(0, 255, 0), font=font)
        display = cv2.cvtColor(np.array(display_pil), cv2.COLOR_RGB2BGR)

        self._show_image(display)

    def _show_characters(self):
        """Tile character crops horizontally on the char canvas."""
        self.char_canvas.delete("all")
        char_imgs = self.result.get("char_imgs", [])
        if not char_imgs:
            self.char_canvas.create_text(140, 35, text="(无字符)", fill="gray")
            return

        canv_w = self.char_canvas.winfo_width() or 300
        char_size = 48
        spacing = 8
        total_w = len(char_imgs) * (char_size + spacing) - spacing
        start_x = max(5, (canv_w - total_w) // 2)

        for i, img in enumerate(char_imgs):
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            if img.ndim == 2:
                h, w = img.shape
            else:
                h, w = img.shape[:2]

            # Resize to fit
            scale = char_size / max(h, w, 1)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized = cv2.resize(img, (new_w, new_h))

            # Convert to PIL and then PhotoImage
            pil_img = Image.fromarray(resized)
            photo = ImageTk.PhotoImage(pil_img)

            cx = start_x + i * (char_size + spacing)
            cy = 35
            self.char_canvas.create_image(cx, cy, image=photo, anchor=tk.CENTER)
            # Keep reference
            setattr(self, f"_char_photo_{i}", photo)

            # Position label
            self.char_canvas.create_text(cx, cy + char_size // 2 + 10,
                                         text=str(i), font=("Consolas", 8), fill="gray")

    def _show_plate_roi(self):
        """Show the plate region on the plate canvas."""
        self.plate_canvas.delete("all")
        plate_img = self.result.get("plate_img")
        if plate_img is None:
            return

        canv_w = self.plate_canvas.winfo_width() or 300
        h, w = plate_img.shape[:2]
        scale = min(50 / h, (canv_w - 10) / w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        if len(plate_img.shape) == 3:
            display = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        else:
            display = plate_img

        pil_img = Image.fromarray(display).resize((new_w, new_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)
        cx = max(5 + new_w // 2, canv_w // 2)
        self.plate_canvas.create_image(cx, 30, image=photo, anchor=tk.CENTER)
        self._plate_photo = photo

    def _show_image(self, img_bgr):
        """Display an OpenCV BGR image on the main canvas."""
        self.image_canvas.delete("all")

        h, w = img_bgr.shape[:2]
        canv_w = self.image_canvas.winfo_width() or 800
        canv_h = self.image_canvas.winfo_height() or 500

        # Scale to fit
        scale = min(canv_w / w, canv_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize((new_w, new_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)

        cx = canv_w // 2
        cy = canv_h // 2
        self.image_canvas.create_image(cx, cy, image=photo, anchor=tk.CENTER)
        self._main_photo = photo  # keep reference

    def _clear(self):
        self.current_image = None
        self.result = None
        self.result_var.set("—")
        self.image_canvas.delete("all")
        self.char_canvas.delete("all")
        self.plate_canvas.delete("all")
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)
        self.detail_text.config(state=tk.DISABLED)
        self.status_label.config(text="就绪 — 请打开图片")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = PlateRecognitionApp()
    app.run()
