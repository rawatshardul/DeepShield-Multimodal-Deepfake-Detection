from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

from image_predictor import ImageDeepfakePredictor
from video_detector import VideoDeepfakeDetector
from audio_predictor import AudioDeepfakePredictor

app = FastAPI(title="DeepGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads')
FRONTEND_DIR  = os.path.join(os.path.dirname(__file__), '..', 'frontend')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'weights')

print("Loading all models...")
image_predictor = ImageDeepfakePredictor(
    model_path=os.path.join(WEIGHTS_DIR, 'best_image_model.pth'))
video_detector  = VideoDeepfakeDetector(
    model_path=os.path.join(WEIGHTS_DIR, 'best_image_model.pth'),
    frames_to_analyze=20)
audio_predictor = AudioDeepfakePredictor(
    model_path=os.path.join(WEIGHTS_DIR, 'best_audio_model.pth'))
print("✅ All models loaded!\n")


# ── Serve frontend ──────────────────────────────────────
@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))

@app.get("/app")
def serve_app():
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))


# ── Health ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "online", "models_loaded": True}


# ── Detection endpoints ─────────────────────────────────
def save_upload(file: UploadFile) -> str:
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return path


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    allowed = ['.jpg', '.jpeg', '.png', '.webp']
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Invalid type '{ext}'. Allowed: {allowed}")
    path = save_upload(file)
    try:
        result = image_predictor.predict(path)
        result['filename'] = file.filename
        result['file_type'] = 'image'
        return result
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    allowed = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Invalid type '{ext}'. Allowed: {allowed}")
    path = save_upload(file)
    try:
        result = video_detector.analyze_video(path)
        result['filename'] = file.filename
        result['file_type'] = 'video'
        return result
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.post("/detect/audio")
async def detect_audio(file: UploadFile = File(...)):
    allowed = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.mp4']
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Invalid type '{ext}'. Allowed: {allowed}")
    path = save_upload(file)
    try:
        # MP4 se audio extract karo
        if ext == '.mp4':
            from moviepy.editor import VideoFileClip
            wav_path = path.replace('.mp4', '.wav')
            clip = VideoFileClip(path)
            clip.audio.write_audiofile(wav_path, logger=None)
            clip.close()
            result = audio_predictor.predict(wav_path)
            if os.path.exists(wav_path):
                os.remove(wav_path)
        else:
            result = audio_predictor.predict(path)
        result['filename'] = file.filename
        result['file_type'] = 'audio'
        return result
    finally:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    print("Starting DeepGuard server...")
    print("Open http://127.0.0.1:8000 in your browser")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)