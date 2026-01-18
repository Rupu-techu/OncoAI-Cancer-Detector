import os
from PIL import Image
import numpy as np


# CONFIGURATION

DATA_DIR = "data/images_binary"
IMG_SIZE = (224, 224)
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


# COUNTERS

processed = 0
skipped = 0

print("Preprocessing started...\n")


# PROCESS IMAGES

for label in ["benign", "malignant"]:
    class_dir = os.path.join(DATA_DIR, label)

    if not os.path.isdir(class_dir):
        print(f"Missing folder: {class_dir}")
        continue

    print(f"Processing class: {label}")

    for file_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, file_name)

        # Skip non-image files
        if not file_name.lower().endswith(VALID_EXTENSIONS):
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMG_SIZE)

            img_array = np.array(img) / 255.0  

            processed += 1

            # Progress update
            if processed % 500 == 0:
                print(f"{processed} images processed...")

        except Exception as e:
            print(f"Skipped corrupted image: {img_path}")
            skipped += 1
            continue

print("\nPreprocessing completed!")
print(f"Total images processed: {processed}")
print(f"Total images skipped: {skipped}")
