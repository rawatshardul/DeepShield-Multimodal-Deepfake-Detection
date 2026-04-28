# ============================================================
#   DEEPGUARD — DGX B200 TRAINING SCRIPT
#   Run this on NVIDIA DGX B200 at GEU
#   Expected training time: 10-15 minutes
# ============================================================

# ── Step 1: Install dependencies ────────────────────────────
import subprocess
subprocess.run(['pip', 'install', 'torch', 'torchvision',
                'efficientnet_pytorch', 'facenet-pytorch',
                'opencv-python', 'librosa', 'soundfile',
                'scikit-learn', 'matplotlib', 'tqdm'], check=True)

# ── Step 2: Imports ──────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

print("="*60)
print("  DEEPGUARD TRAINING ON NVIDIA DGX B200")
print("="*60)
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("="*60)


# ── Step 3: Configuration ────────────────────────────────────
CONFIG = {
    'dataset_path'  : './datasets/image_dataset',
    'weights_dir'   : './models/weights',
    'epochs'        : 15,
    'batch_size'    : 128,      # Large batch size for DGX
    'learning_rate' : 0.001,
    'max_train'     : 60000,    # Use 60k images for training
    'max_val'       : 10000,    # Use 10k images for validation
    'num_workers'   : 8,        # DGX has many CPU cores
    'image_size'    : 224,
}

os.makedirs(CONFIG['weights_dir'], exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}\n")


# ── Step 4: Dataset ──────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, split='train', max_per_class=None):
        self.transform = train_transforms if split == 'train' else val_transforms
        self.images = []
        self.labels = []

        # Support multiple folder name variations
        split_names = {
            'train'      : ['train'],
            'val'        : ['val', 'valid', 'validation'],
        }
        names = split_names.get(split, [split])
        split_dir = None
        for name in names:
            candidate = os.path.join(root_dir, name)
            if os.path.exists(candidate):
                split_dir = candidate
                break

        if split_dir is None:
            print(f"WARNING: Could not find split folder for '{split}'")
            return

        for label, folder in [(0, 'real'), (1, 'fake')]:
            folder_path = os.path.join(split_dir, folder)
            if not os.path.exists(folder_path):
                print(f"WARNING: {folder_path} not found")
                continue
            imgs = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg','.jpeg','.png'))]
            if max_per_class:
                imgs = random.sample(imgs, min(max_per_class, len(imgs)))
            for img in imgs:
                self.images.append(os.path.join(folder_path, img))
                self.labels.append(label)

        real_count = self.labels.count(0)
        fake_count = self.labels.count(1)
        print(f"  {split:10s}: {real_count:6d} real + {fake_count:6d} fake = {len(self.images):6d} total")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


# ── Step 5: Model ────────────────────────────────────────────
class DeepfakeImageDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.efficientnet_b0(weights='IMAGENET1K_V1')
        num_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base(x)


# ── Step 6: Training function ────────────────────────────────
def train():
    print("Loading dataset...")
    train_ds = DeepfakeDataset(
        CONFIG['dataset_path'], 'train', CONFIG['max_train']//2)
    val_ds   = DeepfakeDataset(
        CONFIG['dataset_path'], 'val',   CONFIG['max_val']//2)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              shuffle=True,  num_workers=CONFIG['num_workers'],
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=CONFIG['num_workers'],
                              pin_memory=True)

    print(f"\nLoading EfficientNet-B0 model...")
    model     = DeepfakeImageDetector().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    best_val_acc = 0.0

    print(f"\nStarting training for {CONFIG['epochs']} epochs...\n")

    for epoch in range(CONFIG['epochs']):
        # Training
        model.train()
        train_loss, correct, total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct    += ((outputs > 0.5).float() == labels).sum().item()
            total      += labels.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                             'acc': f'{100*correct/total:.1f}%'})

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]  "):
                images  = images.to(device)
                labels  = labels.to(device).unsqueeze(1)
                outputs = model(images)
                val_loss    += criterion(outputs, labels).item()
                val_correct += ((outputs > 0.5).float() == labels).sum().item()
                val_total   += labels.size(0)

        # Metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)
        train_acc      = 100 * correct     / total
        val_acc        = 100 * val_correct / val_total

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"\nEpoch [{epoch+1}/{CONFIG['epochs']}]")
        print(f"  Train — Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  Val   — Loss: {avg_val_loss:.4f}   | Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                      os.path.join(CONFIG['weights_dir'], 'best_image_model.pth'))
            print(f"  ✅ Best model saved! Val Acc: {val_acc:.2f}%")

        scheduler.step()
        print()

    # Save history
    with open(os.path.join(CONFIG['weights_dir'],
                           'image_training_history.json'), 'w') as f:
        json.dump(history, f)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['train_loss'], label='Train', color='blue')
    ax1.plot(history['val_loss'],   label='Val',   color='red')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch')
    ax1.legend(); ax1.grid(True)
    ax2.plot(history['train_acc'], label='Train', color='blue')
    ax2.plot(history['val_acc'],   label='Val',   color='red')
    ax2.set_title('Accuracy (%)'); ax2.set_xlabel('Epoch')
    ax2.legend(); ax2.grid(True)
    plt.suptitle(f'DeepGuard Training — Best Val Acc: {best_val_acc:.2f}%')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['weights_dir'], 'training_plot.png'), dpi=150)
    print(f"\n📊 Training plot saved!")
    print(f"🎉 Training complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    return history


if __name__ == '__main__':
    train()