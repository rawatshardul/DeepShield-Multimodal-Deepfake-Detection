import cv2
import torch
import numpy as np
from PIL import Image
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from image_predictor import ImageDeepfakePredictor


class VideoDeepfakeDetector:
    def __init__(self, model_path=None, frames_to_analyze=20):
        """
        frames_to_analyze: how many frames to sample from the video
        More frames = more accurate but slower
        20 frames is a good balance
        """
        print("Loading video deepfake detector...")
        self.predictor = ImageDeepfakePredictor(model_path=model_path)
        self.frames_to_analyze = frames_to_analyze

        # Try to load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✅ Video detector ready!")

    def extract_frames(self, video_path):
        """
        Extracts evenly spaced frames from a video.
        Returns a list of PIL Images.
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        print(f"  Video info: {total_frames} frames, "
              f"{fps:.1f} FPS, {duration:.1f} seconds")

        # Calculate which frames to extract
        # We spread them evenly across the video
        if total_frames <= self.frames_to_analyze:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(
                0, total_frames - 1,
                self.frames_to_analyze,
                dtype=int
            ).tolist()

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR (OpenCV) to RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)

        cap.release()
        print(f"  Extracted {len(frames)} frames for analysis")
        return frames, total_frames, fps, duration

    def detect_face(self, pil_image):
        """
        Detects face in image and returns cropped face.
        If no face found, returns the original image.
        """
        # Convert to numpy for OpenCV
        img_array = np.array(pil_image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) > 0:
            # Get the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face

            # Add padding around face (20%)
            padding = int(0.2 * min(w, h))
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(pil_image.width, x + w + padding)
            y2 = min(pil_image.height, y + h + padding)

            face_crop = pil_image.crop((x1, y1, x2, y2))
            return face_crop, True
        else:
            # No face found, use full frame
            return pil_image, False

    def analyze_video(self, video_path):
        """
        Main function — analyzes a video and returns deepfake verdict.

        Returns a dict with full analysis results.
        """
        print(f"\nAnalyzing video: {os.path.basename(video_path)}")
        print("-" * 40)

        if not os.path.exists(video_path):
            return {
                'label': 'ERROR',
                'confidence': 0,
                'status': f'Video file not found: {video_path}'
            }

        try:
            # Step 1: Extract frames
            print("Step 1: Extracting frames...")
            frames, total_frames, fps, duration = self.extract_frames(video_path)

            if len(frames) == 0:
                return {
                    'label': 'ERROR',
                    'confidence': 0,
                    'status': 'Could not extract frames from video'
                }

            # Step 2: Analyze each frame
            print("Step 2: Analyzing frames for deepfakes...")
            frame_results = []
            faces_found = 0

            for i, frame in enumerate(frames):
                # Detect and crop face
                face_img, face_detected = self.detect_face(frame)
                if face_detected:
                    faces_found += 1

                # Run image model on this frame
                result = self.predictor.predict_from_pil(face_img)
                frame_results.append(result)

                if (i + 1) % 5 == 0:
                    print(f"  Analyzed {i+1}/{len(frames)} frames...")

            # Step 3: Aggregate results
            print("Step 3: Computing final verdict...")

            fake_probs = [r['fake_probability'] for r in frame_results
                         if r['status'] == 'success']
            real_probs = [r['real_probability'] for r in frame_results
                         if r['status'] == 'success']

            if len(fake_probs) == 0:
                return {
                    'label': 'ERROR',
                    'confidence': 0,
                    'status': 'All frames failed analysis'
                }

            # Average probability across all frames
            avg_fake_prob = np.mean(fake_probs)
            avg_real_prob = np.mean(real_probs)

            # Count frame votes
            fake_votes = sum(1 for r in frame_results
                           if r.get('label') == 'FAKE')
            real_votes = sum(1 for r in frame_results
                           if r.get('label') == 'REAL')

            # Final verdict
            label = 'FAKE' if avg_fake_prob > 50 else 'REAL'
            confidence = avg_fake_prob if label == 'FAKE' else avg_real_prob

            result = {
                'label': label,
                'confidence': round(confidence, 2),
                'fake_probability': round(avg_fake_prob, 2),
                'real_probability': round(avg_real_prob, 2),
                'frames_analyzed': len(frame_results),
                'fake_votes': fake_votes,
                'real_votes': real_votes,
                'faces_found': faces_found,
                'total_frames': total_frames,
                'fps': round(fps, 2),
                'duration_seconds': round(duration, 2),
                'status': 'success'
            }

            return result

        except Exception as e:
            return {
                'label': 'ERROR',
                'confidence': 0,
                'status': f'Error during analysis: {str(e)}'
            }


# Test the video detector
def create_test_video():
    """Creates a small test video with random frames"""
    import cv2
    import numpy as np

    output_path = '../datasets/test_video.mp4'
    os.makedirs('../datasets', exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 24, (224, 224))

    for i in range(48):  # 2 seconds at 24fps
        # Create a frame with a simple face-like pattern
        frame = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
        # Draw a circle to simulate a face
        cv2.circle(frame, (112, 112), 60, (200, 160, 120), -1)
        cv2.circle(frame, (90, 100), 8, (50, 50, 50), -1)   # left eye
        cv2.circle(frame, (134, 100), 8, (50, 50, 50), -1)  # right eye
        cv2.ellipse(frame, (112, 130), (25, 12), 0, 0, 180,
                    (150, 80, 80), -1)  # mouth
        out.write(frame)

    out.release()
    print(f"✅ Test video created: {output_path}")


# Test the video detector
if __name__ == "__main__":
    print("=" * 50)
    print("  TESTING VIDEO DEEPFAKE DETECTOR")
    print("=" * 50)

    # First create a test video
    print("\nCreating a test video...")
    create_test_video()

    # Now test the detector
    detector = VideoDeepfakeDetector(
        model_path='weights/best_image_model.pth',
        frames_to_analyze=10
    )

    result = detector.analyze_video('../datasets/test_video.mp4')

    print("\n" + "=" * 40)
    print("  VIDEO ANALYSIS RESULT")
    print("=" * 40)
    print(f"  Verdict      : {result['label']}")
    print(f"  Confidence   : {result['confidence']}%")
    print(f"  Fake prob    : {result['fake_probability']}%")
    print(f"  Real prob    : {result['real_probability']}%")

    if result['status'] == 'success':
        print(f"  Frames       : {result['frames_analyzed']} analyzed")
        print(f"  Fake votes   : {result['fake_votes']}")
        print(f"  Real votes   : {result['real_votes']}")
        print(f"  Faces found  : {result['faces_found']}")
        print(f"  Duration     : {result['duration_seconds']}s")

    print(f"  Status       : {result['status']}")
    print("\n✅ Video detector pipeline working!")