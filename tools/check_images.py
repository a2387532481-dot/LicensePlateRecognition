"""Check available images for license plates."""
import cv2, os
base = r"D:\University\PythonProject2"
for f in ["my_test_01.jpg", "my_test_02.jpg", "image.jpg", "test.jpg"]:
    path = os.path.join(base, f)
    img = cv2.imread(path)
    if img is not None:
        print(f"{f}: {img.shape}")
    else:
        print(f"{f}: NOT FOUND or unreadable")
