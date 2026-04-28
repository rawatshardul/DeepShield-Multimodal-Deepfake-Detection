# ============================================================
#   Audio Deepfake Detection — Training Script
#   Uses mel spectrograms + Custom CNN
#   Dataset: ASVspoof or WaveFake
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import os
import sys
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import soundfile as sf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from audio_model import get_audio_model


# ── Audio Dataset ────────────────────────────────────────────
class AudioDeepfakeDataset(Dataset):
    """
    Expects this folder structure:
    audio_dataset/
        train/
            real/   ← real human speech (.wav files)
            fake/   ← AI generated speech (.wav files)
        val/
            real/
            fake/
    """
    def __init__(self, root_dir, split='train', max_per_class=None):
        self.root_dir = root_dir
        self.split = split
        self.sample_rate = 16000
        self.duration = 3
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        self.audio_files = []
        self.labels = []

        import random

        for label, folder in [(0, 'real'), (1, 'fake')]:
            folder_path = os.path.join(root_dir, split, folder)
            if not os.path.exists(folder_path):
                print(f"WARNING: {folder_path} not found")
                continue
            files = [f for f in os.listdir(folder_path)
                    if f.lower().endswith(('.wav','.mp3','.flac','.ogg'))]
            if max_per_class:
                files = random.sample(files, min(max_per_class, len(files)))
            for f in files:
                self.audio_files.append(os.path.join(folder_path, f))
                self.labels.append(label)

        real_count = self.labels.count(0)
        fake_count = self.labels.count(1)
        print(f"  {split:8s}: {real_count} real + {fake_count} fake = {len(self.labels)} total")

    def audio_to_melspectrogram(self, audio_path):
        """Converts audio file to mel spectrogram tensor"""
        import cv2
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        target_length = self.sample_rate * self.duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate,
            n_mels=self.n_mels, n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mel_db   = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        mel_resized = cv2.resize(mel_norm, (128, 128))
        return torch.FloatTensor(mel_resized).unsqueeze(0)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        try:
            mel = self.audio_to_melspectrogram(self.audio_files[idx])
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return mel, label
        except Exception as e:
            # Return zeros if file is corrupted
            return torch.zeros(1, 128, 128), torch.tensor(0.0)


# ── Training Function ────────────────────────────────────────
def train_audio_model(dataset_path, epochs=15, batch_size=32, lr=0.001):
    print("=" * 60)
    print("  AUDIO DEEPFAKE DETECTION — TRAINING")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load dataset
    print("\nLoading audio dataset...")
    train_ds = AudioDeepfakeDataset(dataset_path, 'train', max_per_class=5000)
    val_ds   = AudioDeepfakeDataset(dataset_path, 'val',   max_per_class=1000)

    if len(train_ds) == 0:
        print("❌ No training data found!")
        print("Please add audio files to:", dataset_path)
        return None, None

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                             shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                             shuffle=False, num_workers=0)

    # Model
    model     = get_audio_model().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    best_val_acc = 0.0
    weights_dir  = os.path.join(os.path.dirname(__file__), 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    print(f"\nStarting training for {epochs} epochs...\n")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, correct, total = 0, 0, 0

        for mel, labels in tqdm(train_loader,
                                desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            mel    = mel.to(device)
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(mel)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct    += ((outputs > 0.5).float() == labels).sum().item()
            total      += labels.size(0)

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for mel, labels in tqdm(val_loader,
                                    desc=f"Epoch {epoch+1}/{epochs} [Val]  "):
                mel     = mel.to(device)
                labels  = labels.to(device).unsqueeze(1)
                outputs = model(mel)
                val_loss    += criterion(outputs, labels).item()
                val_correct += ((outputs>0.5).float()==labels).sum().item()
                val_total   += labels.size(0)

        # Metrics
        t_loss = train_loss / len(train_loader)
        v_loss = val_loss   / len(val_loader)
        t_acc  = 100 * correct     / total
        v_acc  = 100 * val_correct / val_total

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"  Train — Loss: {t_loss:.4f} | Acc: {t_acc:.2f}%")
        print(f"  Val   — Loss: {v_loss:.4f} | Acc: {v_acc:.2f}%")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(),
                      os.path.join(weights_dir, 'best_audio_model.pth'))
            print(f"  ✅ Best audio model saved! Val Acc: {v_acc:.2f}%")

        scheduler.step()

    # Save history and plot
    with open(os.path.join(weights_dir, 'audio_training_history.json'), 'w') as f:
        json.dump(history, f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train', color='blue')
    ax1.plot(history['val_loss'],   label='Val',   color='red')
    ax1.set_title('Audio Model Loss')
    ax1.legend(); ax1.grid(True)
    ax2.plot(history['train_acc'], label='Train', color='blue')
    ax2.plot(history['val_acc'],   label='Val',   color='red')
    ax2.set_title('Audio Model Accuracy (%)')
    ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(weights_dir, 'audio_training_plot.png'))
    print(f"\n📊 Audio training plot saved!")
    print(f"🎉 Audio training complete! Best Val Acc: {best_val_acc:.2f}%")
    return model, history


def create_synthetic_audio_dataset():
    """
    Creates a small synthetic audio dataset for testing.
    Replace with real ASVspoof dataset for production.
    """
    import soundfile as sf
    print("Creating synthetic audio dataset for testing...")

    folders = [
        '../datasets/audio_dataset/train/real',
        '../datasets/audio_dataset/train/fake',
        '../datasets/audio_dataset/val/real',
        '../datasets/audio_dataset/val/fake',
    ]
    for f in folders:
        os.makedirs(f, exist_ok=True)

    sr = 16000
    duration = 3
    t = np.linspace(0, duration, sr * duration)

    # Real audio — natural voice simulation
    for i in range(200):
        freq = np.random.uniform(100, 200)
        audio = (0.4 * np.sin(2*np.pi*freq*t) +
                 0.2 * np.sin(2*np.pi*freq*2*t) +
                 0.1 * np.sin(2*np.pi*freq*3*t) +
                 0.05 * np.random.randn(len(t)))
        audio = audio / np.max(np.abs(audio)) * 0.8
        folder = '../datasets/audio_dataset/train/real' if i < 160 \
                 else '../datasets/audio_dataset/val/real'
        sf.write(f'{folder}/real_{i:04d}.wav', audio, sr)

    # Fake audio — synthetic voice simulation
    for i in range(200):
        freq = np.random.uniform(150, 250)
        audio = (0.5 * np.sin(2*np.pi*freq*t) +
                 0.3 * np.sin(2*np.pi*freq*2*t) +
                 0.01 * np.random.randn(len(t)))  # Less natural noise
        audio = audio / np.max(np.abs(audio)) * 0.8
        folder = '../datasets/audio_dataset/train/fake' if i < 160 \
                 else '../datasets/audio_dataset/val/fake'
        sf.write(f'{folder}/fake_{i:04d}.wav', audio, sr)

    print("✅ Synthetic audio dataset created!")
    print("   Train: 160 real + 160 fake")
    print("   Val:    40 real +  40 fake")


if __name__ == "__main__":
    DATASET_PATH = "../datasets/audio_dataset"

    # Create synthetic dataset if no real dataset available
    if not os.path.exists(os.path.join(DATASET_PATH, 'train', 'real')):
        print("No audio dataset found — creating synthetic dataset...")
        create_synthetic_audio_dataset()

    # Train
    train_audio_model(
        dataset_path=DATASET_PATH,
        epochs=10,
        batch_size=16,
        lr=0.001
    )