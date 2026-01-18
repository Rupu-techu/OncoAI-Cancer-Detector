import os
import shutil

# paths
original_dataset = r"C:\Users\debir\OneDrive\Desktop\cancer prediction project\data\images\BreaKHis_v1\BreaKHis_v1\histology_slides\breast"
binary_dataset = r"C:\Users\debir\OneDrive\Desktop\cancer prediction project\data\images_binary"

# create target folders if not exist
os.makedirs(os.path.join(binary_dataset, "benign"), exist_ok=True)
os.makedirs(os.path.join(binary_dataset, "malignant"), exist_ok=True)

# walk through original dataset
for root, dirs, files in os.walk(original_dataset):
    for file in files:
        if file.endswith((".png", ".jpg", ".jpeg")):  # image files
            src_path = os.path.join(root, file)

            # check if path has 'benign' or 'malignant'
            if "benign" in root.lower():
                dest_path = os.path.join(binary_dataset, "benign", file)
            elif "malignant" in root.lower():
                dest_path = os.path.join(binary_dataset, "malignant", file)
            else:
                continue

            # copy file
            shutil.copy2(src_path, dest_path)

print("✅ Images organized into binary dataset (benign/malignant).")
