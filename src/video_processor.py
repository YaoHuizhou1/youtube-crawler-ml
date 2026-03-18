import cv2
import subprocess
import tempfile
import os
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Utility class for video processing operations."""

    @staticmethod
    def extract_frames(video_path: str, timestamps_ms: List[int]) -> List[Tuple[int, any]]:
        """
        Extract frames at specific timestamps.

        Args:
            video_path: Path to video file
            timestamps_ms: List of timestamps in milliseconds

        Returns:
            List of (timestamp_ms, frame) tuples
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []

        try:
            for ts_ms in timestamps_ms:
                frame_number = int((ts_ms / 1000) * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if ret:
                    frames.append((ts_ms, frame))
        finally:
            cap.release()

        return frames

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """Get video metadata."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            return {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration_ms": int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) * 1000
                )
            }
        finally:
            cap.release()

    @staticmethod
    def extract_audio(video_path: str, output_path: str = None) -> str:
        """
        Extract audio from video file.

        Args:
            video_path: Path to video file
            output_path: Optional output path (will generate temp file if not provided)

        Returns:
            Path to extracted audio file
        """
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.wav')

        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            output_path
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise

        return output_path

    @staticmethod
    def create_thumbnail(video_path: str, timestamp_ms: int = 0, output_path: str = None) -> str:
        """
        Create a thumbnail from video.

        Args:
            video_path: Path to video file
            timestamp_ms: Timestamp for thumbnail
            output_path: Optional output path

        Returns:
            Path to thumbnail image
        """
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.jpg')

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int((timestamp_ms / 1000) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret:
                cv2.imwrite(output_path, frame)
                return output_path
            else:
                raise ValueError("Failed to read frame")
        finally:
            cap.release()

    @staticmethod
    def sample_video(video_path: str, sample_duration_s: int = 60, output_path: str = None) -> str:
        """
        Sample a portion of video.

        Args:
            video_path: Path to video file
            sample_duration_s: Duration to sample in seconds
            output_path: Optional output path

        Returns:
            Path to sampled video
        """
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.mp4')

        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-t', str(sample_duration_s),
            '-c', 'copy',
            output_path
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise

        return output_path
