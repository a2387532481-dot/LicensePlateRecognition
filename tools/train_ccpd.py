"""
Train CharCNN models on real CCPD character crops.
Replaces synthetic font-based training with real license plate data.
"""
import os
import sys
import random
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plate_recognition.model import CharCNN, train_model
from plate_recognition.config import (
    DIGIT_MODEL_PATH, LETTER_MODEL_PATH, PROVINCE_MODEL_PATH,
    PROVINCES, LETTERS, DIGITS, CHAR_WIDTH, CHAR_HEIGHT,
)
from torch.utils.data import DataLoader, TensorDataset

CCPD_CHARS_DIR = r"D:\ccpd_train\train\chars"


def load_char_images(chars_dir: str, class_labels: list,
                     max_per_class: int = 0) -> tuple:
    """Load character images from CCPD preprocessed directory.

    Each character class is a subdirectory named by the character itself.
    If max_per_class > 0, randomly downsample classes exceeding the limit.

    Returns (images, labels) as numpy arrays.
    """
    images = []
    labels = []
    label_to_idx = {ch: i for i, ch in enumerate(class_labels)}
    stats = []

    for ch in class_labels:
        ch_dir = os.path.join(chars_dir, ch)
        if not os.path.isdir(ch_dir):
            stats.append((ch, 0, "missing"))
            continue

        pngs = [f for f in os.listdir(ch_dir) if f.lower().endswith(".png")]
        if not pngs:
            stats.append((ch, 0, "empty"))
            continue

        n_avail = len(pngs)
        if max_per_class > 0 and n_avail > max_per_class:
            pngs = random.sample(pngs, max_per_class)
            n_used = max_per_class
        else:
            n_used = n_avail

        for fname in pngs:
            fpath = os.path.join(ch_dir, fname)
            # Use imdecode to avoid cv2.imread encoding issues with Chinese paths
            with open(fpath, 'rb') as f:
                img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (CHAR_WIDTH, CHAR_HEIGHT), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(label_to_idx[ch])

        stats.append((ch, n_used, f"{n_avail} avail"))

    # Print summary
    for ch, n, note in stats:
        if n < 50 and "missing" not in note:
            print(f"  [{ch}] {n} images ({note})")
    missing = [ch for ch, n, note in stats if "missing" in note]
    if missing:
        print(f"  Missing classes: {', '.join(missing)}")

    if not images:
        raise RuntimeError(f"No character images found in {chars_dir}")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    print(f"  Total: {len(images)} images across {len(set(labels))} classes")
    return images, labels


def prepare_ccpd_data(images: np.ndarray, labels: np.ndarray,
                      batch_size: int = 64, test_split: float = 0.2):
    """Convert numpy arrays to PyTorch DataLoaders with stratified-ish split."""
    images = images[:, np.newaxis, :, :]  # add channel dim

    n_test = max(1, int(len(images) * test_split))
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


def train_province_model(chars_dir: str, device: str = "cpu"):
    """Train province classifier on CCPD character crops."""
    print(f"\n{'='*50}")
    print(f"Training PROVINCE classifier ({len(PROVINCES)} classes) on CCPD data")
    print(f"{'='*50}")

    images, labels = load_char_images(chars_dir, PROVINCES, max_per_class=3000)
    train_loader, test_loader = prepare_ccpd_data(images, labels)

    model = CharCNN(num_classes=len(PROVINCES))
    acc = train_model(model, train_loader, test_loader, epochs=30, lr=0.001, device=device)
    torch.save(model.state_dict(), PROVINCE_MODEL_PATH)
    print(f"Province model saved -> {PROVINCE_MODEL_PATH}  (accuracy: {acc:.4f})")
    return acc


def train_letter_model(chars_dir: str, device: str = "cpu"):
    """Train letter classifier on CCPD character crops."""
    print(f"\n{'='*50}")
    print(f"Training LETTER classifier ({len(LETTERS)} classes) on CCPD data")
    print(f"{'='*50}")

    images, labels = load_char_images(chars_dir, LETTERS, max_per_class=5000)
    train_loader, test_loader = prepare_ccpd_data(images, labels)

    model = CharCNN(num_classes=len(LETTERS))
    acc = train_model(model, train_loader, test_loader, epochs=25, lr=0.001, device=device)
    torch.save(model.state_dict(), LETTER_MODEL_PATH)
    print(f"Letter model saved -> {LETTER_MODEL_PATH}  (accuracy: {acc:.4f})")
    return acc


def train_alphanum_model(chars_dir: str, device: str = "cpu"):
    """Train alphanumeric classifier (digits + letters) on CCPD data."""
    alphanum_classes = DIGITS + LETTERS
    print(f"\n{'='*50}")
    print(f"Training ALPHANUM classifier ({len(alphanum_classes)} classes) on CCPD data")
    print(f"{'='*50}")

    images, labels = load_char_images(chars_dir, alphanum_classes, max_per_class=5000)
    train_loader, test_loader = prepare_ccpd_data(images, labels)

    model = CharCNN(num_classes=len(alphanum_classes))
    acc = train_model(model, train_loader, test_loader, epochs=25, lr=0.001, device=device)
    alphanum_path = os.path.join(os.path.dirname(PROVINCE_MODEL_PATH), "alphanum_cnn.pth")
    torch.save(model.state_dict(), alphanum_path)
    print(f"Alphanum model saved -> {alphanum_path}  (accuracy: {acc:.4f})")
    return acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading CCPD character crops from: {CCPD_CHARS_DIR}\n")

    if not os.path.isdir(CCPD_CHARS_DIR):
        print(f"ERROR: {CCPD_CHARS_DIR} not found.")
        print("Run preprocess_ccpd.py first:")
        print("  python tools/preprocess_ccpd.py --ccpd_dir D:/CCPD2019/CCPD2019"
              " --out_dir D:/ccpd_train --max 20000 --val_split 0.15 --chars_only")
        sys.exit(1)

    p_acc = train_province_model(CCPD_CHARS_DIR, device)
    l_acc = train_letter_model(CCPD_CHARS_DIR, device)
    a_acc = train_alphanum_model(CCPD_CHARS_DIR, device)

    print(f"\n{'='*50}")
    print("CCPD Training Complete!")
    print(f"  Province accuracy:  {p_acc:.4f}")
    print(f"  Letter accuracy:    {l_acc:.4f}")
    print(f"  Alphanum accuracy:  {a_acc:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
