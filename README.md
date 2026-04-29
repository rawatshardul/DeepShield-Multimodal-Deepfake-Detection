# 🛡️ DeepShield — Multimodal Deepfake Detection System

DeepShield is an AI-powered deepfake detection system that identifies manipulated content across **images, videos, and audio** using state-of-the-art deep learning models.

---

## 🚀 Features

- 🖼️ **Image Detection** — Detects AI-generated or manipulated images with 97.77% accuracy
- 🎥 **Video Detection** — Analyzes video frames to detect deepfake videos
- 🎵 **Audio Detection** — Identifies AI-generated or cloned voices with 99%+ accuracy
- 🌐 **Web Interface** — Clean, modern UI for easy file upload and analysis
- ⚡ **Fast API Backend** — Built with FastAPI for high performance

---

## 🧠 Models

| Modality | Architecture | Accuracy |
|----------|-------------|----------|
| Image | EfficientNet-B0 | 97.77% |
| Video | EfficientNet-B0 (frame analysis) | — |
| Audio | Custom CNN (Mel-Spectrogram) | 99%+ |

---

## 📦 Dataset

| Dataset | Samples |
|---------|---------|
| Image (Real) | 89,788 |
| Image (Fake) | 84,037 |
| Audio (FakeAVCeleb v1.2) | 21,560 clips |

Models trained on **NVIDIA DGX B200** GPU.

---

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, PyTorch
- **Frontend:** HTML, CSS, JavaScript
- **Models:** EfficientNet-B0, Custom CNN
- **Audio Processing:** Librosa, Mel-Spectrogram
- **Training Hardware:** NVIDIA DGX B200 (45GB GPU)

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/anushkabishtgithub/DeepShield-Multimodal-Deepfake-Detection.git
cd DeepShield-Multimodal-Deepfake-Detection
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add model weights
Download the trained weights and place them in:
models/weights/best_image_model.pth
models/weights/best_audio_model.pth

### 5. Run the server
```bash
cd backend
python main.py
```

### 6. Open in browser
http://127.0.0.1:8000

---

## 📁 Project Structure
DeepShield/
├── backend/
│   └── main.py              # FastAPI server
├── frontend/
│   └── index.html           # Web interface
├── models/
│   ├── image_model.py       # EfficientNet model
│   ├── image_predictor.py   # Image inference
│   ├── audio_model.py       # CNN audio model
│   ├── audio_predictor.py   # Audio inference
│   ├── video_detector.py    # Video inference
│   └── weights/             # Trained model weights
├── static/
│   └── uploads/             # Temporary uploads
└── requirements.txt

---

## 🎯 Usage

1. Open `http://127.0.0.1:8000` in your browser
2. Select the type of media — **Image, Video, or Audio**
3. Upload your file
4. Get instant **REAL / FAKE** prediction with confidence score

---

## 👩‍💻 Developer

**Shardul Rawat**
Graphic Era Hill University, Dehradun
B.Tech Computer Science Engineering

---
