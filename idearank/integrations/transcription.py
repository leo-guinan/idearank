"""Video transcription using yt-dlp and faster-whisper.

Implements a two-tier strategy:
1. Try to extract native YouTube subtitles (instant, free)
2. Fall back to downloading audio and using faster-whisper (local, GPU-accelerated)
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Literal
import shutil

logger = logging.getLogger(__name__)

# Optional imports
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("faster-whisper not installed. Install with: pip install faster-whisper")

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False
    logger.warning("yt-dlp not installed. Install with: pip install yt-dlp")


class TranscriptionClient:
    """Client for transcribing YouTube videos using yt-dlp and faster-whisper."""
    
    def __init__(
        self,
        whisper_model: str = "small",
        device: Literal["cpu", "cuda", "auto"] = "auto",
        compute_type: str = "int8",
    ):
        """Initialize transcription client.
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            device: Device to run Whisper on (cpu, cuda, or auto)
            compute_type: Computation type (int8, float16, float32)
        """
        if not YTDLP_AVAILABLE:
            raise ImportError(
                "yt-dlp not installed. "
                "Install with: pip install yt-dlp"
            )
        
        self.whisper_model_name = whisper_model
        self.device = device
        self.compute_type = compute_type
        self._whisper_model = None  # Lazy load
    
    @property
    def whisper_model(self) -> Optional[WhisperModel]:
        """Lazy-load Whisper model when needed."""
        if self._whisper_model is None and WHISPER_AVAILABLE:
            logger.info(f"Loading Whisper model: {self.whisper_model_name}")
            self._whisper_model = WhisperModel(
                self.whisper_model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._whisper_model
    
    def transcribe_video(
        self,
        video_id: str,
        prefer_subtitles: bool = True,
    ) -> tuple[Optional[str], str]:
        """Transcribe a YouTube video.
        
        Strategy:
        1. Try to extract native YouTube subtitles (fast, free)
        2. Fall back to downloading audio and using faster-whisper
        
        Args:
            video_id: YouTube video ID
            prefer_subtitles: Try to extract native subtitles first
            
        Returns:
            (transcript_text, source) where source is "youtube-subs" or "whisper"
        """
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Step 1: Try to extract native subtitles
        if prefer_subtitles:
            logger.info(f"Attempting to extract subtitles for {video_id}...")
            subtitle_text = self._extract_subtitles(video_url)
            if subtitle_text:
                logger.info(f"✓ Got native subtitles for {video_id}")
                return subtitle_text, "youtube-subs"
        
        # Step 2: Fall back to Whisper transcription
        logger.info(f"No subtitles found. Transcribing with Whisper for {video_id}...")
        
        if not WHISPER_AVAILABLE:
            logger.error("faster-whisper not available. Cannot transcribe.")
            return None, "none"
        
        transcript_text = self._transcribe_with_whisper(video_url)
        if transcript_text:
            logger.info(f"✓ Transcribed {video_id} with Whisper")
            return transcript_text, "whisper"
        
        logger.warning(f"Failed to transcribe {video_id}")
        return None, "none"
    
    def _extract_subtitles(self, video_url: str) -> Optional[str]:
        """Extract native YouTube subtitles using yt-dlp.
        
        Tries to get English subtitles (auto-generated or manual).
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Subtitle text or None if not available
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # yt-dlp options for subtitle extraction
            ydl_opts = {
                'skip_download': True,
                'writeautomaticsub': True,
                'writesubtitles': True,
                'subtitleslangs': ['en', 'en-US', 'en-GB'],
                'subtitlesformat': 'vtt',
                'outtmpl': str(tmpdir_path / 'video.%(id)s'),
                'quiet': True,
                'no_warnings': True,
            }
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                
                # Find the subtitle file
                vtt_files = list(tmpdir_path.glob('*.vtt'))
                if not vtt_files:
                    return None
                
                # Read and parse the VTT file
                subtitle_text = self._parse_vtt(vtt_files[0])
                return subtitle_text
                
            except Exception as e:
                logger.debug(f"Subtitle extraction failed: {e}")
                return None
    
    def _parse_vtt(self, vtt_path: Path) -> str:
        """Parse VTT subtitle file and extract text.
        
        Args:
            vtt_path: Path to VTT file
            
        Returns:
            Plain text transcript
        """
        with open(vtt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # VTT format:
        # WEBVTT
        # 
        # 00:00:00.000 --> 00:00:02.000
        # Subtitle text here
        
        text_lines = []
        for line in lines:
            line = line.strip()
            # Skip headers, timestamps, and empty lines
            if line.startswith('WEBVTT') or '-->' in line or not line:
                continue
            # Skip cue identifiers (numbers/IDs)
            if line.isdigit():
                continue
            text_lines.append(line)
        
        return ' '.join(text_lines)
    
    def _transcribe_with_whisper(self, video_url: str) -> Optional[str]:
        """Download audio and transcribe with faster-whisper.
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Transcript text or None if failed
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Step 1: Download audio using yt-dlp
            audio_path = self._download_audio(video_url, tmpdir_path)
            if not audio_path:
                return None
            
            # Step 2: Transcribe with faster-whisper
            try:
                logger.info(f"Transcribing audio with Whisper ({self.whisper_model_name})...")
                
                segments, info = self.whisper_model.transcribe(
                    str(audio_path),
                    task="transcribe",
                    language="en",  # Can be removed for auto-detection
                    beam_size=5,
                    vad_filter=True,  # Voice activity detection
                )
                
                # Collect all segments
                text_segments = [segment.text.strip() for segment in segments]
                transcript_text = ' '.join(text_segments)
                
                logger.info(f"✓ Transcription complete: {len(transcript_text)} chars")
                return transcript_text
                
            except Exception as e:
                logger.error(f"Whisper transcription failed: {e}")
                return None
    
    def _download_audio(self, video_url: str, output_dir: Path) -> Optional[Path]:
        """Download audio from YouTube video.
        
        Args:
            video_url: YouTube video URL
            output_dir: Directory to save audio
            
        Returns:
            Path to downloaded audio file or None if failed
        """
        output_template = str(output_dir / 'audio.%(id)s.%(ext)s')
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                video_id = info['id']
                
                # Find the downloaded audio file
                audio_files = list(output_dir.glob(f'audio.{video_id}.mp3'))
                if not audio_files:
                    logger.error("Audio file not found after download")
                    return None
                
                logger.info(f"✓ Downloaded audio: {audio_files[0].name}")
                return audio_files[0]
                
        except Exception as e:
            logger.error(f"Audio download failed: {e}")
            return None
    
    @staticmethod
    def check_dependencies() -> dict[str, bool]:
        """Check if required dependencies are available.
        
        Returns:
            Dict with availability status for each dependency
        """
        status = {
            'yt-dlp': YTDLP_AVAILABLE,
            'faster-whisper': WHISPER_AVAILABLE,
            'ffmpeg': shutil.which('ffmpeg') is not None,
        }
        return status
    
    @staticmethod
    def print_dependency_status():
        """Print status of all dependencies."""
        status = TranscriptionClient.check_dependencies()
        
        print("Transcription Dependencies:")
        print("-" * 40)
        for dep, available in status.items():
            status_icon = "✓" if available else "✗"
            print(f"  {status_icon} {dep}")
        print()
        
        if not all(status.values()):
            print("Missing dependencies:")
            if not status['yt-dlp']:
                print("  pip install yt-dlp")
            if not status['faster-whisper']:
                print("  pip install faster-whisper")
            if not status['ffmpeg']:
                print("  Install ffmpeg: https://ffmpeg.org/download.html")

