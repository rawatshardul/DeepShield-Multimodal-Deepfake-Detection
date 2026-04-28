import torch
import cv2
import librosa
import fastapi
import torchvision
import numpy
import pandas
import sklearn
import PIL

print("✅ PyTorch:", torch.__version__)
print("✅ OpenCV:", cv2.__version__)
print("✅ Librosa:", librosa.__version__)
print("✅ FastAPI:", fastapi.__version__)
print("✅ Torchvision:", torchvision.__version__)
print("✅ Numpy:", numpy.__version__)
print("✅ Pandas:", pandas.__version__)
print("✅ Scikit-learn:", sklearn.__version__)
print("✅ Pillow:", PIL.__version__)
print("✅ GPU Available:", torch.cuda.is_available())
print("\n🎉 Phase 1 Complete! All packages working perfectly!")