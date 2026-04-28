import torch
import torch.nn as nn
from torchvision import models

class DeepfakeImageDetector(nn.Module):
    def __init__(self):
        super(DeepfakeImageDetector, self).__init__()

        # Load pretrained EfficientNet-B0
        # This is already trained on millions of images
        # We just replace the last layer for our task
        self.base = models.efficientnet_b0(weights='IMAGENET1K_V1')

        # Get the number of features from the last layer
        num_features = self.base.classifier[1].in_features

        # Replace the classifier with our own
        # Output: 1 value (0 = real, 1 = fake)
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x):
        return self.base(x)


def get_model():
    """Returns the model ready for training"""
    model = DeepfakeImageDetector()
    return model


# Test the model
if __name__ == "__main__":
    print("Testing model architecture...")
    model = get_model()

    # Create a fake batch of 4 images (3 channels, 224x224 pixels)
    dummy_input = torch.randn(4, 3, 224, 224)

    # Pass through model
    output = model(dummy_input)

    print(f"✅ Input shape:  {dummy_input.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Output values (should be between 0-1): {output.detach().numpy().flatten()}")
    print(f"✅ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\n🎉 Image model architecture working perfectly!")