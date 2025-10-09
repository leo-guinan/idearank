"""YouTube data fetching and transcription.

Fetches video metadata, transcripts, and handles transcription when needed.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import logging
import re

logger = logging.getLogger(__name__)

# Optional imports
try:
    from googleapiclient.discovery import build
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False
    logger.warning("YouTube libraries not installed. Install with: pip install -e '.[youtube]'")

# Gladia is not available via pip, would need custom integration
GLADIA_AVAILABLE = False


@dataclass
class YouTubeVideoData:
    """Raw data from YouTube API."""
    
    video_id: str
    channel_id: str
    title: str
    description: str
    published_at: datetime
    
    # Statistics
    view_count: int
    like_count: int
    comment_count: int
    
    # Duration in seconds
    duration_seconds: int
    
    # Transcript (if available)
    transcript: Optional[str] = None
    transcript_source: str = "none"  # "youtube", "gladia", "none"
    
    # Tags and categories
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class YouTubeClient:
    """Client for fetching YouTube channel and video data."""
    
    def __init__(
        self,
        youtube_api_key: Optional[str] = None,
        gladia_api_key: Optional[str] = None,
    ):
        """Initialize YouTube client.
        
        Args:
            youtube_api_key: YouTube Data API v3 key (optional, uses quota)
            gladia_api_key: Gladia API key for transcription (optional)
        """
        if not YOUTUBE_AVAILABLE:
            raise ImportError(
                "YouTube libraries not installed. "
                "Install with: pip install -e '.[youtube]'"
            )
        
        self.youtube_api_key = youtube_api_key
        self.gladia_api_key = gladia_api_key
        
        # Initialize YouTube API client if key provided
        if youtube_api_key:
            self.youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        else:
            self.youtube = None
            logger.warning(
                "No YouTube API key provided. "
                "Only public data and transcripts will be available."
            )
    
    def extract_channel_id(self, channel_url: str) -> str:
        """Extract channel ID from various YouTube URL formats.
        
        Supports:
        - youtube.com/@username
        - youtube.com/channel/CHANNEL_ID
        - youtube.com/c/CustomName
        """
        # Handle @username format
        if '@' in channel_url:
            username = channel_url.split('@')[1].split('/')[0]
            return self._resolve_username_to_channel_id(username)
        
        # Handle /channel/ format
        if '/channel/' in channel_url:
            return channel_url.split('/channel/')[1].split('/')[0]
        
        # Handle /c/ format
        if '/c/' in channel_url:
            custom_name = channel_url.split('/c/')[1].split('/')[0]
            return self._resolve_custom_name_to_channel_id(custom_name)
        
        raise ValueError(f"Could not parse channel URL: {channel_url}")
    
    def _resolve_username_to_channel_id(self, username: str) -> str:
        """Resolve @username to channel ID using YouTube API."""
        if not self.youtube:
            raise ValueError(
                "YouTube API key required to resolve @username. "
                "Please provide youtube_api_key."
            )
        
        # Search for the channel by username
        request = self.youtube.search().list(
            part="snippet",
            q=username,
            type="channel",
            maxResults=1
        )
        response = request.execute()
        
        if not response.get('items'):
            raise ValueError(f"Channel not found: @{username}")
        
        channel_id = response['items'][0]['snippet']['channelId']
        logger.info(f"Resolved @{username} to channel ID: {channel_id}")
        return channel_id
    
    def _resolve_custom_name_to_channel_id(self, custom_name: str) -> str:
        """Resolve custom name to channel ID."""
        if not self.youtube:
            raise ValueError("YouTube API key required to resolve custom names.")
        
        # Similar to username resolution
        request = self.youtube.search().list(
            part="snippet",
            q=custom_name,
            type="channel",
            maxResults=1
        )
        response = request.execute()
        
        if not response.get('items'):
            raise ValueError(f"Channel not found: {custom_name}")
        
        return response['items'][0]['snippet']['channelId']
    
    def get_channel_videos(
        self,
        channel_id: str,
        max_results: int = 50,
    ) -> List[str]:
        """Get list of video IDs from a channel.
        
        Args:
            channel_id: YouTube channel ID
            max_results: Maximum number of videos to fetch
            
        Returns:
            List of video IDs
        """
        if not self.youtube:
            raise ValueError("YouTube API key required to fetch channel videos.")
        
        video_ids = []
        
        # Get uploads playlist ID
        request = self.youtube.channels().list(
            part="contentDetails",
            id=channel_id
        )
        response = request.execute()
        
        if not response.get('items'):
            raise ValueError(f"Channel not found: {channel_id}")
        
        uploads_playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        # Fetch videos from uploads playlist
        next_page_token = None
        while len(video_ids) < max_results:
            request = self.youtube.playlistItems().list(
                part="contentDetails",
                playlistId=uploads_playlist_id,
                maxResults=min(50, max_results - len(video_ids)),
                pageToken=next_page_token
            )
            response = request.execute()
            
            for item in response['items']:
                video_ids.append(item['contentDetails']['videoId'])
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        
        logger.info(f"Found {len(video_ids)} videos in channel {channel_id}")
        return video_ids
    
    def get_video_data(self, video_id: str) -> YouTubeVideoData:
        """Fetch complete data for a single video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            YouTubeVideoData with metadata and transcript
        """
        # Get video metadata
        if self.youtube:
            metadata = self._get_video_metadata_api(video_id)
        else:
            metadata = self._get_video_metadata_basic(video_id)
        
        # Try to get transcript
        transcript, source = self._get_video_transcript(video_id)
        
        return YouTubeVideoData(
            video_id=video_id,
            channel_id=metadata['channel_id'],
            title=metadata['title'],
            description=metadata['description'],
            published_at=metadata['published_at'],
            view_count=metadata['view_count'],
            like_count=metadata.get('like_count', 0),
            comment_count=metadata.get('comment_count', 0),
            duration_seconds=metadata['duration_seconds'],
            transcript=transcript,
            transcript_source=source,
            tags=metadata.get('tags', []),
        )
    
    def _get_video_metadata_api(self, video_id: str) -> dict:
        """Fetch video metadata using YouTube API."""
        request = self.youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id
        )
        response = request.execute()
        
        if not response.get('items'):
            raise ValueError(f"Video not found: {video_id}")
        
        item = response['items'][0]
        snippet = item['snippet']
        statistics = item['statistics']
        content_details = item['contentDetails']
        
        # Parse duration (ISO 8601 format like PT15M30S)
        duration_seconds = self._parse_duration(content_details['duration'])
        
        return {
            'channel_id': snippet['channelId'],
            'title': snippet['title'],
            'description': snippet['description'],
            'published_at': datetime.fromisoformat(snippet['publishedAt'].replace('Z', '+00:00')),
            'view_count': int(statistics.get('viewCount', 0)),
            'like_count': int(statistics.get('likeCount', 0)),
            'comment_count': int(statistics.get('commentCount', 0)),
            'duration_seconds': duration_seconds,
            'tags': snippet.get('tags', []),
        }
    
    def _get_video_metadata_basic(self, video_id: str) -> dict:
        """Fetch basic metadata without API (limited info)."""
        # This is a fallback - very limited info
        logger.warning(f"Using basic metadata for {video_id} (no API key)")
        return {
            'channel_id': 'unknown',
            'title': f'Video {video_id}',
            'description': '',
            'published_at': datetime.now(),
            'view_count': 0,
            'duration_seconds': 0,
        }
    
    def _parse_duration(self, duration_str: str) -> int:
        """Parse ISO 8601 duration to seconds."""
        # Example: PT15M30S -> 930 seconds
        match = re.match(
            r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?',
            duration_str
        )
        if not match:
            return 0
        
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        
        return hours * 3600 + minutes * 60 + seconds
    
    def _get_video_transcript(self, video_id: str) -> tuple[Optional[str], str]:
        """Get video transcript from YouTube or Gladia.
        
        Returns:
            (transcript_text, source)
        """
        # Try YouTube's auto-generated or uploaded transcripts first
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = ' '.join([entry['text'] for entry in transcript_list])
            logger.info(f"Got YouTube transcript for {video_id}")
            return transcript_text, "youtube"
        except Exception as e:
            logger.warning(f"YouTube transcript not available for {video_id}: {e}")
        
        # Try Gladia transcription as fallback
        if self.gladia_api_key and GLADIA_AVAILABLE:
            try:
                transcript_text = self._transcribe_with_gladia(video_id)
                logger.info(f"Transcribed {video_id} with Gladia")
                return transcript_text, "gladia"
            except Exception as e:
                logger.error(f"Gladia transcription failed for {video_id}: {e}")
        
        logger.warning(f"No transcript available for {video_id}")
        return None, "none"
    
    def _transcribe_with_gladia(self, video_id: str) -> str:
        """Transcribe video using Gladia API.
        
        Note: This is a simplified implementation.
        Real implementation would need to download audio, upload to Gladia, etc.
        """
        # TODO: Implement Gladia transcription
        # Steps would be:
        # 1. Get video URL
        # 2. Download audio (using yt-dlp or similar)
        # 3. Upload to Gladia
        # 4. Wait for transcription
        # 5. Return text
        
        logger.warning("Gladia transcription not yet implemented")
        raise NotImplementedError("Gladia transcription coming soon")
    
    def get_channel_data(
        self,
        channel_url: str,
        max_videos: int = 50,
    ) -> List[YouTubeVideoData]:
        """Fetch all data for a channel.
        
        Args:
            channel_url: YouTube channel URL
            max_videos: Maximum number of videos to fetch
            
        Returns:
            List of YouTubeVideoData objects
        """
        logger.info(f"Fetching channel data from: {channel_url}")
        
        # Extract channel ID
        channel_id = self.extract_channel_id(channel_url)
        logger.info(f"Channel ID: {channel_id}")
        
        # Get video IDs
        video_ids = self.get_channel_videos(channel_id, max_videos)
        logger.info(f"Fetching data for {len(video_ids)} videos...")
        
        # Fetch data for each video
        videos = []
        for i, video_id in enumerate(video_ids, 1):
            logger.info(f"Processing video {i}/{len(video_ids)}: {video_id}")
            try:
                video_data = self.get_video_data(video_id)
                videos.append(video_data)
            except Exception as e:
                logger.error(f"Failed to fetch video {video_id}: {e}")
                continue
        
        logger.info(f"Successfully fetched {len(videos)} videos")
        return videos

