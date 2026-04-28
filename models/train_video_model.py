# ============================================================
#   Video Deepfake Detection — Training Script
#   NOTE: Video detection uses the Image Model (EfficientNet)
#   Training the image model automatically trains video detection
#   This script is a wrapper that calls image model training
# ============================================================

import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_image_model import train_model
from video_detector import VideoDeepfakeDetector


def train_video_model(dataset_path, epochs=15, batch_size=32):
    """
    Video deepfake detection uses the same EfficientNet model
    as image detection. Training on face images automatically
    enables video detection since videos are analyzed
    frame-by-frame using the same model.
    """
    print("=" * 60)
    print("  VIDEO DEEPFAKE DETECTION — TRAINING")
    print("=" * 60)
    print("""
    How video detection works:
    ┌─────────────────────────────────────────┐
    │  Video File                             │
    │       ↓                                 │
    │  Extract 20 evenly spaced frames        │
    │       ↓                                 │
    │  Detect & crop face in each frame       │
    │       ↓                                 │
    │  Run EfficientNet (Image Model)         │
    │  on each frame                          │
    │       ↓                                 │
    │  Average probabilities across frames    │
    │       ↓                                 │
    │  Final verdict: REAL or FAKE            │
    └─────────────────────────────────────────┘

    Training the IMAGE model = Training the VIDEO detector
    """)

    print("Starting image model training for video detection...\n")

    # Train the image model — this is what powers video detection
    model, history = train_model(
        dataset_path=dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=0.001
    )

    print("\n✅ Video detector is now trained!")
    print("   Weights saved to: models/weights/best_image_model.pth")
    print("   These weights are used by video_detector.py automatically")

    return model, history


def test_video_detector():
    """Tests the video detector with the test video"""
    print("\n" + "="*50)
    print("  TESTING VIDEO DETECTOR")
    print("="*50)

    weights_path = os.path.join(
        os.path.dirname(__file__), 'weights', 'best_image_model.pth'
    )

    detector = VideoDeepfakeDetector(
        model_path=weights_path,
        frames_to_analyze=20
    )

    test_video = '../datasets/test_video.mp4'
    if not os.path.exists(test_video):
        print("Creating test video...")
        from video_detector import create_test_video
        create_test_video()

    print(f"\nTesting on: {test_video}")
    result = detector.analyze_video(test_video)

    print("\n" + "="*40)
    print("  VIDEO TEST RESULT")
    print("="*40)
    print(f"  Verdict      : {result['label']}")
    print(f"  Confidence   : {result['confidence']}%")
    print(f"  Frames       : {result.get('frames_analyzed', 'N/A')}")
    print(f"  Status       : {result['status']}")


if __name__ == "__main__":
    DATASET_PATH = "../datasets/image_dataset"

    # Train
    train_video_model(
        dataset_path=DATASET_PATH,
        epochs=15,
        batch_size=32
    )

    # Test
    test_video_detector()