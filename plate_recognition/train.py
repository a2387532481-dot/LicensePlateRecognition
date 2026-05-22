"""
Train character classifiers on synthetic data (digits, letters, provinces).
Synthetic data uses font rendering similar to actual license plate characters.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plate_recognition.model import (
    CharCNN, generate_synthetic_chars,
    train_model, prepare_data,
)
from plate_recognition.config import (
    DIGIT_MODEL_PATH, LETTER_MODEL_PATH, PROVINCE_MODEL_PATH,
    DIGITS, LETTERS, PROVINCES, CHAR_WIDTH, CHAR_HEIGHT,
)


def train_digit_model(device: str = "cpu"):
    """Train digit classifier (0-9) on MNIST for robust real-world recognition."""
    print("=" * 50)
    print("Training DIGIT classifier (0-9) on MNIST…")
    print("=" * 50)

    from plate_recognition.model import load_mnist
    import cv2

    train_images, train_labels, test_images, test_labels = load_mnist()

    # Normalize to [0, 1]
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # Resize to standard size
    train_resized = np.array([cv2.resize(img, (CHAR_WIDTH, CHAR_HEIGHT)) for img in train_images])
    test_resized = np.array([cv2.resize(img, (CHAR_WIDTH, CHAR_HEIGHT)) for img in test_images])

    # Add channel dimension
    train_resized = train_resized[:, np.newaxis, :, :]
    test_resized = test_resized[:, np.newaxis, :, :]

    from torch.utils.data import DataLoader, TensorDataset
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_resized).float(), torch.from_numpy(train_labels).long()),
        batch_size=64, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_resized).float(), torch.from_numpy(test_labels).long()),
        batch_size=64, shuffle=False,
    )

    model = CharCNN(num_classes=len(DIGITS))
    acc = train_model(model, train_loader, test_loader, epochs=10, device=device)
    torch.save(model.state_dict(), DIGIT_MODEL_PATH)
    print(f"Digit model saved → {DIGIT_MODEL_PATH}  (accuracy: {acc:.4f})")
    return acc


def train_letter_model(device: str = "cpu"):
    """Train letter classifier (A-Z, excluding I/O) on synthetic data."""
    print("\n" + "=" * 50)
    print(f"Training LETTER classifier ({len(LETTERS)} classes) on synthetic data…")
    print("=" * 50)

    images, labels = generate_synthetic_chars(
        LETTERS, len(LETTERS), samples_per_class=600, use_chinese_font=False,
    )
    train_loader, test_loader = prepare_data(images, labels)

    model = CharCNN(num_classes=len(LETTERS))
    acc = train_model(model, train_loader, test_loader, epochs=30, device=device)
    torch.save(model.state_dict(), LETTER_MODEL_PATH)
    print(f"Letter model saved → {LETTER_MODEL_PATH}  (accuracy: {acc:.4f})")
    return acc


def train_province_model(device: str = "cpu"):
    """Train Chinese province classifier (31 classes) on synthetic data."""
    print("\n" + "=" * 50)
    print(f"Training PROVINCE classifier ({len(PROVINCES)} classes) on synthetic data…")
    print("=" * 50)

    images, labels = generate_synthetic_chars(
        PROVINCES, len(PROVINCES), samples_per_class=600, use_chinese_font=True,
    )
    train_loader, test_loader = prepare_data(images, labels)

    model = CharCNN(num_classes=len(PROVINCES))
    acc = train_model(model, train_loader, test_loader, epochs=25, device=device)
    torch.save(model.state_dict(), PROVINCE_MODEL_PATH)
    print(f"Province model saved → {PROVINCE_MODEL_PATH}  (accuracy: {acc:.4f})")
    return acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    digit_acc = train_digit_model(device)
    letter_acc = train_letter_model(device)
    province_acc = train_province_model(device)

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"  Digit accuracy:    {digit_acc:.4f}")
    print(f"  Letter accuracy:   {letter_acc:.4f}")
    print(f"  Province accuracy: {province_acc:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
