import os
import numpy as np
from PIL import Image
import random

print("Creating sample dataset for testing...")

# Create folder structure
folders = [
    '../datasets/image_dataset/train/real',
    '../datasets/image_dataset/train/fake',
    '../datasets/image_dataset/val/real',
    '../datasets/image_dataset/val/fake',
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

def create_fake_face(path, label, idx):
    """Creates a synthetic face-like image for testing"""
    img_array = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)

    # Add some structure to make it look slightly different
    if label == 'real':
        # Warmer tones for real
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] + 30, 0, 255)
    else:
        # Cooler tones for fake
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] + 30, 0, 255)

    img = Image.fromarray(img_array)
    img.save(path)

# Create training images
print("Creating training images...")
for i in range(100):
    create_fake_face(
        f'../datasets/image_dataset/train/real/real_{i:04d}.jpg',
        'real', i
    )
    create_fake_face(
        f'../datasets/image_dataset/train/fake/fake_{i:04d}.jpg',
        'fake', i
    )

# Create validation images
print("Creating validation images...")
for i in range(30):
    create_fake_face(
        f'../datasets/image_dataset/val/real/real_{i:04d}.jpg',
        'real', i
    )
    create_fake_face(
        f'../datasets/image_dataset/val/fake/fake_{i:04d}.jpg',
        'fake', i
    )

print("✅ Sample dataset created!")
print("   Training: 100 real + 100 fake = 200 images")
print("   Validation: 30 real + 30 fake = 60 images")
print("\nNOTE: These are synthetic images for testing only.")
print("For the real project, replace with actual deepfake dataset.")