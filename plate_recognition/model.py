"""
CNN model for character classification and training utilities.
"""
import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image, ImageDraw, ImageFont
from .config import (
    CHAR_WIDTH, CHAR_HEIGHT, DIGIT_MODEL_PATH,
    LETTER_MODEL_PATH, PROVINCE_MODEL_PATH,
    PROVINCES, LETTERS, DIGITS,
)


class CharCNN(nn.Module):
    """Lightweight CNN for 28x28 character classification."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 3x3
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── MNIST loading ──────────────────────────────────────────────────

def load_mnist() -> tuple:
    """Load MNIST dataset from the raw IDX files, return (images, labels)."""
    def read_idx(filename):
        with open(filename, "rb") as f:
            zero, dtype, dims = struct.unpack(">HBB", f.read(4))
            shape = tuple(struct.unpack(">I", f.read(4))[0] for _ in range(dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

    train_images = read_idx(os.path.join(MNIST_DIR, "train-images-idx3-ubyte"))
    train_labels = read_idx(os.path.join(MNIST_DIR, "train-labels-idx1-ubyte"))
    test_images = read_idx(os.path.join(MNIST_DIR, "t10k-images-idx3-ubyte"))
    test_labels = read_idx(os.path.join(MNIST_DIR, "t10k-labels-idx1-ubyte"))

    return train_images, train_labels, test_images, test_labels


# ── Synthetic data generation ──────────────────────────────────────

def _get_available_fonts() -> list:
    """Return paths to available system fonts."""
    font_dirs = [
        r"C:\Windows\Fonts",
        "/usr/share/fonts",
        "/System/Library/Fonts",
    ]
    fonts = []
    for d in font_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith((".ttf", ".otf", ".ttc")):
                    fonts.append(os.path.join(d, f))
    return fonts


def _pick_chinese_font(fonts: list) -> str:
    """Pick a suitable Chinese font from available fonts."""
    preferred = ["simhei", "msyh", "simsun", "kaiti", "stxihei",
                 "notosanscjk", "wqy", "wenquan"]
    for font in fonts:
        fn = os.path.basename(font).lower()
        for p in preferred:
            if p in fn:
                return font
    # Fallback: any ttf
    for font in fonts:
        if font.lower().endswith(".ttf"):
            return font
    return None


def generate_synthetic_chars(
    characters: list,
    num_classes: int,
    samples_per_class: int = 200,
    use_chinese_font: bool = False,
) -> tuple:
    """
    Generate synthetic character images for training.
    Returns (images, labels) as numpy arrays.
    """
    fonts = _get_available_fonts()
    if not fonts:
        raise RuntimeError("No fonts found for synthetic data generation")

    if use_chinese_font:
        font_path = _pick_chinese_font(fonts)
    else:
        # Pick a standard English font
        font_path = None
        for f in fonts:
            fn = os.path.basename(f).lower()
            if "arial" in fn or "dejavu" in fn:
                font_path = f
                break
        if font_path is None:
            font_path = fonts[0]

    if font_path is None:
        raise RuntimeError("No suitable font found")

    images = []
    labels = []

    for class_idx, char in enumerate(characters):
        for _ in range(samples_per_class):
            img = _render_char(
                char, font_path,
                size_variation=True,
                rotation_variation=True,
                noise=True,
                translation=True,
            )
            images.append(img)
            labels.append(class_idx)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # Shuffle
    idx = np.random.permutation(len(images))
    return images[idx], labels[idx]


def _render_char(
    char: str,
    font_path: str,
    size_variation: bool = True,
    rotation_variation: bool = True,
    noise: bool = True,
    translation: bool = True,
) -> np.ndarray:
    """Render a character at high resolution and downscale for clean, thick glyphs."""
    import cv2

    # Render at large size and downscale for clean, dense characters
    large_w, large_h = 128, 128
    font_size = 112  # Fill ~88% of canvas height

    if size_variation:
        font_size += np.random.randint(-8, 9)

    if len(char) > 1:  # Chinese character — needs to be larger for complex strokes
        font_size = int(font_size * 0.78)

    # PIL rendering
    pil_img = Image.new("L", (large_w, large_h), color=0)
    draw = ImageDraw.Draw(pil_img)

    try:
        if font_path.lower().endswith(".ttc"):
            font = ImageFont.truetype(font_path, font_size, index=0)
        else:
            font = ImageFont.truetype(font_path, font_size)
    except OSError:
        font = ImageFont.load_default()

    if translation:
        dx = np.random.randint(-10, 11)
        dy = np.random.randint(-10, 11)
    else:
        dx, dy = 0, 0

    draw.text((large_w // 2 + dx, large_h // 2 + dy), char,
              fill=255, font=font, anchor="mm")

    # Rotation
    if rotation_variation and np.random.random() > 0.5:
        angle = np.random.uniform(-12, 12)
        pil_img = pil_img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)

    # Convert to numpy
    img = np.array(pil_img, dtype=np.float32)

    # Blur and resize — keep continuous values (no binarization)
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    img = cv2.resize(img, (CHAR_WIDTH, CHAR_HEIGHT), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0

    # Add noise
    if noise and np.random.random() > 0.3:
        nl = np.random.uniform(0.005, 0.04)
        img += np.random.normal(0, nl, img.shape)
        img = np.clip(img, 0, 1)

    return img.astype(np.float32)


# ── OpenCV helpers (avoid import at module level to prevent circular imports) ──

def cv2_resize(img: np.ndarray, size: tuple) -> np.ndarray:
    import cv2
    return cv2.resize(img, size)


def cv2_erode(img: np.ndarray, k: int) -> np.ndarray:
    import cv2
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(img, kernel, iterations=1)


def cv2_dilate(img: np.ndarray, k: int) -> np.ndarray:
    import cv2
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


# ── Training ───────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 15,
    lr: float = 0.001,
    device: str = "cpu",
) -> float:
    """Train the CNN and return test accuracy."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / max(total, 1)
    return accuracy


def prepare_data(images: np.ndarray, labels: np.ndarray, batch_size: int = 64, test_split: float = 0.15):
    """Convert numpy arrays to PyTorch DataLoaders."""
    # Add channel dimension
    images = images.reshape(-1, 1, CHAR_HEIGHT, CHAR_WIDTH if images.ndim == 3 else CHAR_WIDTH)
    if images.ndim == 3:
        images = images[:, np.newaxis, :, :]

    # Split
    n_test = int(len(images) * test_split)
    idx = np.random.permutation(len(images))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    train_data = TensorDataset(
        torch.from_numpy(images[train_idx]).float(),
        torch.from_numpy(labels[train_idx]).long(),
    )
    test_data = TensorDataset(
        torch.from_numpy(images[test_idx]).float(),
        torch.from_numpy(labels[test_idx]).long(),
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
