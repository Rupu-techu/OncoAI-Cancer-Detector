import os
import numpy as np
from PIL import Image

DATA_DIR = "data/images_binary"
IMG_SIZE = (128, 128)  
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


X = []
y = []

label_map = {
    "benign": 0,
    "malignant": 1
}

print("Saving dataset started...\n")


for label in ["benign", "malignant"]:
    class_dir = os.path.join(DATA_DIR, label)

    if not os.path.isdir(class_dir):
        print(f"Missing folder: {class_dir}")
        continue

    print(f"Processing class: {label}")

    for file_name in os.listdir(class_dir):

        
        if not file_name.lower().endswith(VALID_EXTENSIONS):
            continue

        img_path = os.path.join(class_dir, file_name)

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMG_SIZE)

            img_array = np.array(img, dtype=np.float32) / 255.0

            X.append(img_array)
            y.append(label_map[label])

            
            if len(X) % 500 == 0:
                print(f"{len(X)} images saved...")

        except Exception as e:
            print(f"Skipped corrupted image: {img_path}")
            continue


X = np.array(X)
y = np.array(y)

np.save("X_images.npy", X)
np.save("y_labels.npy", y)

print("\nDataset saved successfully!")
print("X shape:", X.shape)
print("y shape:", y.shape)
