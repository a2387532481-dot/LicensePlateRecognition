import cv2
import numpy as np

for car_name in ['car.jpg']:
    for i in range(7):
        raw_path = f'debug_slot_{car_name}_0_{i}_raw.jpg'
        ocr_path = f'debug_slot_{car_name}_0_{i}_ocr_input.jpg'
        bin_path = f'debug_slot_{car_name}_0_{i}_binary.jpg'

        raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        ocr = cv2.imread(ocr_path, cv2.IMREAD_GRAYSCALE)
        bin_img = cv2.imread(bin_path, cv2.IMREAD_GRAYSCALE)

        if raw is None:
            print(f'Slot {i}: raw is None')
            continue

        print(f'\nSlot {i}:')
        print(f'  Raw shape={raw.shape}, mean={raw.mean():.1f}, min={raw.min()}, max={raw.max()}')
        print(f'  Text-mask mean={bin_img.mean():.1f}, white_pct={bin_img.mean()/255*100:.1f}%')
        print(f'  OCR input shape={ocr.shape}, mean={ocr.mean():.1f}')

        # Check if character is visible in OCR input
        if ocr.mean() > 250:
            print(f'  WARNING: OCR input is nearly all white!')
        elif ocr.mean() < 5:
            print(f'  WARNING: OCR input is nearly all black!')
