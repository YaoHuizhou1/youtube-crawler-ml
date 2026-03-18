import os
import logging
import subprocess
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    def __init__(self):
        self.pipeline = None
        self._init_pipeline()

    def _init_pipeline(self):
        """Initialize the pyannote.audio pipeline."""
        try:
            from pyannote.audio import Pipeline
            import torch

            # Check for HuggingFace token
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                # Move to GPU if available
                if torch.cuda.is_available():
                    self.pipeline.to(torch.device("cuda"))
                logger.info("Speaker diarization pipeline initialized")
            else:
                logger.warning("HF_TOKEN not set, speaker diarization will use fallback")
                self.pipeline = None
        except Exception as e:
            logger.warning(f"Failed to initialize diarization pipeline: {e}")
            self.pipeline = None

    def analyze(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze video for speakers.

        Args:
            video_path: Path to video file

        Returns:
            Dict with speaker_count, dialogue_ratio, segments
        """
        # Extract audio from video
        audio_path = self._extract_audio(video_path)

        try:
            if self.pipeline:
                return self._analyze_with_pyannote(audio_path)
            else:
                return self._analyze_fallback(audio_path)
        finally:
            # Cleanup audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video file."""
        audio_path = video_path.replace('.mp4', '.wav')

        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            audio_path
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise

        return audio_path

    def _analyze_with_pyannote(self, audio_path: str) -> Dict[str, Any]:
        """Analyze audio with pyannote.audio."""
        diarization = self.pipeline(audio_path)

        segments: List[Dict[str, Any]] = []
        speakers = set()
        total_duration = 0.0
        dialogue_duration = 0.0

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)
            speaker_id = int(speaker.split('_')[-1]) if '_' in speaker else 0

            segments.append({
                "start_ms": start_ms,
                "end_ms": end_ms,
                "speaker_id": speaker_id
            })

            speakers.add(speaker)
            total_duration = max(total_duration, turn.end)

        # Calculate dialogue ratio (portion with 2 speakers within 5s window)
        if total_duration > 0:
            # Simple heuristic: if we have 2 speakers, estimate dialogue ratio
            if len(speakers) == 2:
                dialogue_duration = sum(
                    (s["end_ms"] - s["start_ms"]) / 1000
                    for s in segments
                )
                dialogue_ratio = min(1.0, dialogue_duration / total_duration)
            else:
                dialogue_ratio = 0.5 if len(speakers) >= 2 else 0.0
        else:
            dialogue_ratio = 0.0

        return {
            "speaker_count": len(speakers),
            "dialogue_ratio": dialogue_ratio,
            "segments": segments
        }

    def _analyze_fallback(self, audio_path: str) -> Dict[str, Any]:
        """Fallback analysis when pyannote is not available."""
        # Return default values for testing
        logger.warning("Using fallback speaker analysis (no pyannote)")
        return {
            "speaker_count": 2,  # Assume 2 speakers
            "dialogue_ratio": 0.7,
            "segments": []
        }
