"""End-to-end pipeline: YouTube → IdeaRank scores.

Fetches YouTube data, transcribes, embeds, indexes, and scores.
"""

from typing import List, Optional
from datetime import datetime
import logging

from idearank.models import Video, Channel
from idearank.config import IdeaRankConfig
from idearank.pipeline import IdeaRankPipeline
from idearank.integrations.youtube import YouTubeClient, YouTubeVideoData
from idearank.integrations.storage import SQLiteStorage

logger = logging.getLogger(__name__)


class YouTubePipeline:
    """Complete pipeline from YouTube URL to IdeaRank scores."""
    
    def __init__(
        self,
        idearank_pipeline: IdeaRankPipeline,
        youtube_client: YouTubeClient,
        storage: SQLiteStorage,
        config: Optional[IdeaRankConfig] = None,
    ):
        """Initialize YouTube pipeline.
        
        Args:
            idearank_pipeline: Configured IdeaRankPipeline with providers
            youtube_client: YouTubeClient for data fetching
            storage: SQLiteStorage for persistence
            config: IdeaRankConfig (if not already in idearank_pipeline)
        """
        self.idearank_pipeline = idearank_pipeline
        self.youtube_client = youtube_client
        self.storage = storage
        self.config = config or IdeaRankConfig.default()
    
    def process_channel(
        self,
        channel_url: str,
        max_videos: int = 50,
        force_refresh: bool = False,
    ) -> tuple[Channel, dict]:
        """Process entire YouTube channel.
        
        Args:
            channel_url: YouTube channel URL (e.g., youtube.com/@username)
            max_videos: Maximum number of videos to process
            force_refresh: If True, re-fetch even if in database
            
        Returns:
            (Channel object, scores_dict)
        """
        logger.info(f"=" * 70)
        logger.info(f"Processing YouTube channel: {channel_url}")
        logger.info(f"=" * 70)
        
        # Step 1: Fetch YouTube data
        logger.info("\n[1/6] Fetching YouTube data...")
        youtube_videos = self.youtube_client.get_channel_data(
            channel_url,
            max_videos=max_videos,
        )
        logger.info(f"✓ Fetched {len(youtube_videos)} videos")
        
        # Step 2: Convert to IdeaRank Video objects
        logger.info("\n[2/6] Converting to IdeaRank format...")
        videos, channel = self._convert_to_idearank_format(youtube_videos)
        logger.info(f"✓ Converted {len(videos)} videos")
        
        # Step 3: Save to SQLite
        logger.info("\n[3/6] Saving to SQLite database...")
        self.storage.save_channel(channel)
        for i, (video, yt_data) in enumerate(zip(videos, youtube_videos), 1):
            self.storage.save_video(video, yt_data)
            if i % 10 == 0:
                logger.info(f"  Saved {i}/{len(videos)} videos...")
        logger.info(f"✓ Saved {len(videos)} videos to database")
        
        # Step 4: Generate embeddings
        logger.info("\n[4/6] Generating embeddings...")
        self.idearank_pipeline.process_videos_batch(videos)
        logger.info(f"✓ Generated {len(videos)} embeddings")
        
        # Step 5: Index in Chroma
        logger.info("\n[5/6] Indexing embeddings in Chroma Cloud...")
        self.idearank_pipeline.index_videos(videos)
        logger.info(f"✓ Indexed {len(videos)} videos in Chroma")
        
        # Step 6: Compute IdeaRank scores
        logger.info("\n[6/6] Computing IdeaRank scores...")
        scores = self._score_all_videos(videos, channel)
        logger.info(f"✓ Computed {len(scores)} IdeaRank scores")
        
        # Save scores to database
        logger.info("\nSaving scores to database...")
        for video_id, score in scores.items():
            self.storage.save_video_score(video_id, channel.id, score)
        
        # Compute and save channel score
        channel_score = self.idearank_pipeline.channel_scorer.score_channel(
            channel,
            video_scores=scores,
        )
        self.storage.save_channel_score(channel.id, channel_score)
        logger.info(f"✓ Channel score: {channel_score.score:.4f}")
        
        logger.info(f"\n{'=' * 70}")
        logger.info("Pipeline complete!")
        logger.info(f"{'=' * 70}")
        
        return channel, scores
    
    def _convert_to_idearank_format(
        self,
        youtube_videos: List[YouTubeVideoData],
    ) -> tuple[List[Video], Channel]:
        """Convert YouTube data to IdeaRank Video and Channel objects."""
        videos = []
        
        if not youtube_videos:
            raise ValueError("No videos provided")
        
        # Use first video to get channel info
        first_video = youtube_videos[0]
        channel_id = first_video.channel_id
        
        for yt_video in youtube_videos:
            # Estimate analytics if not available
            # Impressions ≈ views * 5 (typical YouTube CTR ~20%)
            impression_count = yt_video.view_count * 5
            
            # Watch time ≈ views * avg_duration
            # Assume avg completion rate of 50% for estimation
            avg_duration_estimate = yt_video.duration_seconds * 0.5
            watch_time = yt_video.view_count * avg_duration_estimate
            
            # Create Video object
            video = Video(
                id=yt_video.video_id,
                channel_id=yt_video.channel_id,
                title=yt_video.title,
                description=yt_video.description,
                transcript=yt_video.transcript or "",
                published_at=yt_video.published_at,
                snapshot_time=datetime.now(),
                # Analytics
                view_count=yt_video.view_count,
                impression_count=impression_count,
                watch_time_seconds=float(watch_time),
                avg_view_duration=float(avg_duration_estimate),
                video_duration=float(yt_video.duration_seconds),
                # Trust signals (would need custom detection)
                has_citations=False,  # TODO: Detect citations in description
                citation_count=0,
                source_diversity_score=0.5,  # Neutral default
                correction_count=0,
                # Tags
                tags=yt_video.tags or [],
            )
            videos.append(video)
        
        # Create Channel object
        channel = Channel(
            id=channel_id,
            name=f"Channel {channel_id}",  # Would get from API
            description="YouTube channel",
            created_at=min(v.published_at for v in videos),
            videos=videos,
        )
        
        return videos, channel
    
    def _score_all_videos(
        self,
        videos: List[Video],
        channel: Channel,
    ) -> dict:
        """Score all videos and return dict of {video_id: IdeaRankScore}."""
        scores = {}
        
        for i, video in enumerate(videos, 1):
            logger.info(f"  Scoring video {i}/{len(videos)}: {video.title[:50]}...")
            try:
                score = self.idearank_pipeline.score_video(
                    video,
                    channel,
                    compute_analytics_context=True,
                )
                scores[video.id] = score
            except Exception as e:
                logger.error(f"  Failed to score {video.id}: {e}")
                continue
        
        return scores
    
    def print_summary(
        self,
        channel: Channel,
        scores: dict,
    ) -> None:
        """Print summary of results."""
        print("\n" + "=" * 70)
        print("IdeaRank Summary")
        print("=" * 70)
        
        print(f"\nChannel: {channel.name}")
        print(f"Videos analyzed: {len(scores)}")
        
        # Sort by score
        sorted_scores = sorted(
            scores.items(),
            key=lambda x: x[1].score,
            reverse=True,
        )
        
        print(f"\nTop 5 videos by IdeaRank:")
        print("-" * 70)
        
        for i, (video_id, score) in enumerate(sorted_scores[:5], 1):
            video = next(v for v in channel.videos if v.id == video_id)
            print(f"\n{i}. {video.title[:60]}")
            print(f"   Score: {score.score:.4f} | Gates: {'✓' if score.passes_gates else '✗'}")
            print(f"   U={score.uniqueness.score:.3f} | "
                  f"C={score.cohesion.score:.3f} | "
                  f"L={score.learning.score:.3f} | "
                  f"Q={score.quality.score:.3f} | "
                  f"T={score.trust.score:.3f}")
        
        print("\n" + "=" * 70)

