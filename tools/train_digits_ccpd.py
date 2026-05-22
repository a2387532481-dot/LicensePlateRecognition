"""
Train digit model (0-9) on real CCPD character crops.
Overwrites plate_recognition/models/digit_cnn.pth.
"""
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plate_recognition.model import CharCNN
from plate_recognition.config import DIGIT_MODEL_PATH, DIGITS, CHAR_WIDTH, CHAR_HEIGHT

CCPD_CHARS_DIR = r"D:\ccpd_train\train\chars"
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DigitDataset(Dataset):
    """Load CCPD digit crops with on-the-fly augmentation matching the
    inference preprocessing: OTSU -> tight crop -> blur -> resize 28x28."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = True):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy()

        if self.augment:
            # 轻微平移 (±2px)
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            if dx != 0 or dy != 0:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # 轻微模糊 (50% 概率)
            if random.random() < 0.5:
                ks = random.choice([3])
                img = cv2.GaussianBlur(img, (ks, ks), 0)

            # 微小亮度变化
            if random.random() < 0.3:
                delta = random.uniform(-15, 15)
                img = np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)

        # Resize 28x28, normalize to [0, 1]
        img = cv2.resize(img, (CHAR_WIDTH, CHAR_HEIGHT), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        # Add channel dim -> (1, 28, 28)
        tensor = torch.from_numpy(img).float().unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label


def load_digit_data(chars_dir: str) -> tuple:
    """Load all digit (0-9) character crops from CCPD preprocessed directory.

    Images are 64x64 binary PNGs that were produced by preprocess_char.
    We load as-is and let the Dataset handle augmentation/resize.
    """
    all_images = []
    all_labels = []

    for digit_str in DIGITS:
        ch_dir = os.path.join(chars_dir, digit_str)
        if not os.path.isdir(ch_dir):
            print(f"  WARNING: '{digit_str}' directory missing: {ch_dir}")
            continue

        pngs = [f for f in os.listdir(ch_dir) if f.lower().endswith(".png")]
        print(f"  [{digit_str}] {len(pngs)} samples")

        for fname in pngs:
            fpath = os.path.join(ch_dir, fname)
            # Use imdecode to avoid cv2.imread Chinese path encoding issues
            with open(fpath, "rb") as f:
                buf = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            all_images.append(img)
            all_labels.append(int(digit_str))

    if not all_images:
        raise RuntimeError(f"No digit images found in {chars_dir}")

    images = np.array(all_images, dtype=np.uint8)
    labels = np.array(all_labels, dtype=np.int64)
    print(f"  Total: {len(images)} digit images across {len(set(labels))} classes")
    return images, labels


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    return running_loss / total, correct / total


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading CCPD digit crops from: {CCPD_CHARS_DIR}")

    images, labels = load_digit_data(CCPD_CHARS_DIR)

    # Train / val split (85/15), stratified by shuffling
    idx = np.random.permutation(len(images))
    n_val = int(len(images) * 0.15)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = DigitDataset(images[train_idx], labels[train_idx], augment=True)
    val_ds = DigitDataset(images[val_idx], labels[val_idx], augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    model = CharCNN(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), DIGIT_MODEL_PATH)

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.4f}")

    print(f"\nBest val accuracy: {best_acc:.4f}")
    print(f"Model saved -> {DIGIT_MODEL_PATH}")


if __name__ == "__main__":
    main()
