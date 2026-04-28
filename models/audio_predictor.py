import torch
import numpy as np
import librosa
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from audio_model import get_audio_model


class AudioDeepfakePredictor:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = get_audio_model().to(self.device)

        self.sample_rate = 16000
        self.duration = 5
        self.n_mels = 128

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            print(f"✅ Loaded audio model weights from {model_path}")
        else:
            print("⚠️  No audio weights found — using untrained model")

        self.model.eval()

    def audio_to_melspectrogram(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        target_length = self.sample_rate * self.duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

        tensor = torch.FloatTensor(mel_db).unsqueeze(0).unsqueeze(0)
        return tensor

    def predict(self, audio_path):
        try:
            if not os.path.exists(audio_path):
                return {
                    'label': 'ERROR',
                    'confidence': 0,
                    'status': f'File not found: {audio_path}'
                }

            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = len(audio) / self.sample_rate

            mel_tensor = self.audio_to_melspectrogram(audio_path)
            mel_tensor = mel_tensor.to(self.device)

            with torch.no_grad():
                output = self.model(mel_tensor)
                fake_prob = output.item()
                real_prob = 1 - fake_prob

            label = 'FAKE' if fake_prob > 0.5 else 'REAL'
            confidence = fake_prob if label == 'FAKE' else real_prob

            return {
                'label': label,
                'confidence': round(confidence * 100, 2),
                'fake_probability': round(fake_prob * 100, 2),
                'real_probability': round(real_prob * 100, 2),
                'duration_seconds': round(duration, 2),
                'sample_rate': sr,
                'status': 'success'
            }

        except Exception as e:
            return {
                'label': 'ERROR',
                'confidence': 0,
                'fake_probability': 0,
                'real_probability': 0,
                'status': f'error: {str(e)}'
            }