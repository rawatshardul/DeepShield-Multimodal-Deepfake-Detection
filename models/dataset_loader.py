import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


# These transforms prepare images for EfficientNet
# Real research papers use exactly these values
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),        # Augmentation
    transforms.RandomRotation(10),            # Augmentation
    transforms.ColorJitter(brightness=0.2,
                           contrast=0.2),     # Augmentation
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],           # ImageNet mean
        std=[0.229, 0.224, 0.225]             # ImageNet std
    )
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class DeepfakeDataset(Dataset):
    """
    Expects this folder structure:
    dataset/
        train/
            real/   <- real face images
            fake/   <- deepfake images
        val/
            real/
            fake/
    """
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        self.transform = train_transforms if split == 'train' else val_transforms
        self.images = []
        self.labels = []

        # Load real images (label = 0)
        real_dir = os.path.join(root_dir, split, 'real')
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(real_dir, img_name))
                    self.labels.append(0)  # 0 = real

        # Load fake images (label = 1)
        fake_dir = os.path.join(root_dir, split, 'fake')
        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(fake_dir, img_name))
                    self.labels.append(1)  # 1 = fake

        print(f"✅ {split} set: {self.labels.count(0)} real, "
              f"{self.labels.count(1)} fake images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load and convert image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


def get_dataloaders(dataset_path, batch_size=32):
    """Returns train and validation dataloaders"""
    train_dataset = DeepfakeDataset(dataset_path, split='train')

    # Support both 'valid' and 'val' folder names
    val_split = 'valid' if os.path.exists(os.path.join(dataset_path, 'valid')) \
    else 'validation' if os.path.exists(os.path.join(dataset_path, 'validation')) \
    else 'val'

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0      # Keep 0 for Windows
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader