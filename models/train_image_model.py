import torch
import torch.nn as nn
import torch.optim as optim
from image_model import get_model
from dataset_loader import get_dataloaders
import matplotlib.pyplot as plt
import os
import json


def train_model(dataset_path, epochs=10, batch_size=32, learning_rate=0.001):

    print("=" * 50)
    print("  DEEPFAKE IMAGE DETECTION - TRAINING")
    print("=" * 50)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\nLoading dataset...")
    train_loader, val_loader = get_dataloaders(dataset_path, batch_size)

    # Load model
    print("Loading model...")
    model = get_model().to(device)

    # Loss function and optimizer
    criterion = nn.BCELoss()   # Binary Cross Entropy for real/fake
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler - reduces LR when stuck
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Track metrics
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    best_val_acc = 0.0

    print("\nStarting training...\n")

    for epoch in range(epochs):
        # ---- TRAINING PHASE ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] "
                      f"Step [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        # ---- VALIDATION PHASE ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"\nEpoch [{epoch+1}/{epochs}] Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('weights', exist_ok=True)
            torch.save(model.state_dict(), 'weights/best_image_model.pth')
            print(f"  ✅ New best model saved! Val Acc: {val_acc:.2f}%")

        scheduler.step()
        print()

    print(f"\n🎉 Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%")

    # Save training history
    with open('weights/image_training_history.json', 'w') as f:
        json.dump(history, f)

    # Plot results
    plot_training(history)
    return model, history


def plot_training(history):
    """Saves training graphs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(history['val_loss'], label='Val Loss', color='red')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train Acc', color='blue')
    ax2.plot(history['val_acc'], label='Val Acc', color='red')
    ax2.set_title('Training & Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs('weights', exist_ok=True)
    plt.savefig('weights/training_plot.png')
    print("📊 Training plot saved to weights/training_plot.png")


if __name__ == "__main__":
    DATASET_PATH = "../datasets/image_dataset"
    train_model(
        dataset_path=DATASET_PATH,
        epochs=10,
        batch_size=16,
        learning_rate=0.001
    )