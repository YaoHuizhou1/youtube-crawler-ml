import cv2
import mediapipe as mp
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class FaceDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Full range model
            min_detection_confidence=0.5
        )
        logger.info("FaceDetector initialized")

    def analyze(self, video_path: str, sample_rate: int = 5) -> Dict[str, Any]:
        """
        Analyze video for faces.

        Args:
            video_path: Path to video file
            sample_rate: Sample every N seconds

        Returns:
            Dict with avg_face_count, max_face_count, min_face_count, frame_results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * sample_rate)

        frame_results: List[Dict[str, int]] = []
        face_counts: List[int] = []
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    timestamp_ms = int((frame_count / fps) * 1000)
                    face_count = self._detect_faces_in_frame(frame)

                    frame_results.append({
                        "timestamp_ms": timestamp_ms,
                        "face_count": face_count
                    })
                    face_counts.append(face_count)

                frame_count += 1

        finally:
            cap.release()

        if not face_counts:
            return {
                "avg_face_count": 0.0,
                "max_face_count": 0,
                "min_face_count": 0,
                "frame_results": []
            }

        return {
            "avg_face_count": sum(face_counts) / len(face_counts),
            "max_face_count": max(face_counts),
            "min_face_count": min(face_counts),
            "frame_results": frame_results
        }

    def _detect_faces_in_frame(self, frame) -> int:
        """Detect faces in a single frame."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.face_detection.process(rgb_frame)

        if results.detections:
            return len(results.detections)
        return 0

    def __del__(self):
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
