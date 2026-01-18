'''This file is ONLY for:
Image counting
Image display
Dataset understanding'''
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

base_dir = "data/images_binary"

# Count images
print("Image count per class:")
for cls in os.listdir(base_dir):
    cls_path = os.path.join(base_dir, cls)
    if os.path.isdir(cls_path):
        print(cls, ":", len(os.listdir(cls_path)))

# Show a random image
class_name = random.choice(os.listdir(base_dir))
class_path = os.path.join(base_dir, class_name)

img_name = random.choice(os.listdir(class_path))
img_path = os.path.join(class_path, img_name)

img = Image.open(img_path)

print("Image size:", img.size)
print("Image mode:", img.mode)

plt.imshow(img)
plt.title(class_name)
plt.axis("off")
plt.show()
