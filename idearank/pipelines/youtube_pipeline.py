"""End-to-end pipeline: YouTube → IdeaRank scores.

Fetches YouTube data, transcribes, embeds, indexes, and scores.
"""

from typing import List, Optional
from datetime import datetime
import logging

from idearank.models import ContentItem, ContentSource
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
        max_videos: Optional[int] = 50,
        force_refresh: bool = False,
    ) -> tuple[ContentSource, dict]:
        """Process entire YouTube channel.
        
        Args:
            channel_url: YouTube channel URL (e.g., youtube.com/@username)
            max_videos: Maximum number of videos to process (None = all videos)
            force_refresh: If True, re-fetch even if in database
            
        Returns:
            (ContentSource object, scores_dict)
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
        
        # Step 2: Convert to IdeaRank ContentItem objects
        logger.info("\n[2/6] Converting to IdeaRank format...")
        content_items, content_source = self._convert_to_idearank_format(youtube_videos)
        logger.info(f"✓ Converted {len(content_items)} videos")
        
        # Step 3: Save to SQLite
        logger.info("\n[3/6] Saving to SQLite database...")
        self.storage.save_content_source(content_source)
        for i, (item, yt_data) in enumerate(zip(content_items, youtube_videos), 1):
            self.storage.save_content_item(item, yt_data)
            if i % 10 == 0:
                logger.info(f"  Saved {i}/{len(content_items)} videos...")
        logger.info(f"✓ Saved {len(content_items)} videos to database")
        
        # Step 4: Generate embeddings
        logger.info("\n[4/6] Generating embeddings...")
        self.idearank_pipeline.process_content_batch(content_items)
        logger.info(f"✓ Generated {len(content_items)} embeddings")
        
        # Step 5: Index in Chroma
        logger.info("\n[5/6] Indexing embeddings in Chroma Cloud...")
        self.idearank_pipeline.index_content(content_items)
        logger.info(f"✓ Indexed {len(content_items)} videos in Chroma")
        
        # Step 6: Compute IdeaRank scores
        logger.info("\n[6/6] Computing IdeaRank scores...")
        scores = self._score_all_content(content_items, content_source)
        logger.info(f"✓ Computed {len(scores)} IdeaRank scores")
        
        # Save scores to database
        logger.info("\nSaving scores to database...")
        for item_id, score in scores.items():
            self.storage.save_content_score(item_id, content_source.id, score)
        
        # Compute and save source score
        source_score = self.idearank_pipeline.source_scorer.score_source(
            content_source,
            content_scores=scores,
        )
        self.storage.save_source_score(content_source.id, source_score)
        logger.info(f"✓ Content source score: {source_score.score:.4f}")
        
        logger.info(f"\n{'=' * 70}")
        logger.info("Pipeline complete!")
        logger.info(f"{'=' * 70}")
        
        return content_source, scores
    
    def _convert_to_idearank_format(
        self,
        youtube_videos: List[YouTubeVideoData],
    ) -> tuple[List[ContentItem], ContentSource]:
        """Convert YouTube data to IdeaRank ContentItem and ContentSource objects."""
        content_items = []
        
        if not youtube_videos:
            raise ValueError("No videos provided")
        
        # Use first video to get channel info
        first_video = youtube_videos[0]
        source_id = first_video.channel_id
        
        for yt_video in youtube_videos:
            # Estimate analytics if not available
            # Impressions ≈ views * 5 (typical YouTube CTR ~20%)
            impression_count = yt_video.view_count * 5
            
            # Watch time ≈ views * avg_duration
            # Assume avg completion rate of 50% for estimation
            avg_duration_estimate = yt_video.duration_seconds * 0.5
            watch_time = yt_video.view_count * avg_duration_estimate
            
            # Create ContentItem object
            content_item = ContentItem(
                id=yt_video.video_id,
                content_source_id=yt_video.channel_id,
                title=yt_video.title,
                description=yt_video.description,
                body=yt_video.transcript or "",  # transcript is the main body for videos
                published_at=yt_video.published_at,
                captured_at=datetime.now(),
                # Analytics
                view_count=yt_video.view_count,
                impression_count=impression_count,
                watch_time_seconds=float(watch_time),
                avg_view_duration=float(avg_duration_estimate),
                content_duration=float(yt_video.duration_seconds),
                # Trust signals (would need custom detection)
                has_citations=False,  # TODO: Detect citations in description
                citation_count=0,
                source_diversity_score=0.5,  # Neutral default
                correction_count=0,
                # Tags
                tags=yt_video.tags or [],
            )
            content_items.append(content_item)
        
        # Create ContentSource object
        content_source = ContentSource(
            id=source_id,
            name=f"YouTube Channel {source_id}",  # Would get from API
            description="YouTube channel",
            created_at=min(item.published_at for item in content_items),
            content_items=content_items,
        )
        
        return content_items, content_source
    
    def _score_all_content(
        self,
        content_items: List[ContentItem],
        content_source: ContentSource,
    ) -> dict:
        """Score all content items and return dict of {item_id: IdeaRankScore}."""
        scores = {}
        
        for i, item in enumerate(content_items, 1):
            logger.info(f"  Scoring video {i}/{len(content_items)}: {item.title[:50]}...")
            try:
                score = self.idearank_pipeline.score_content_item(
                    item,
                    content_source,
                    compute_analytics_context=True,
                )
                scores[item.id] = score
            except Exception as e:
                logger.error(f"  Failed to score {item.id}: {e}")
                continue
        
        return scores
    
    def print_summary(
        self,
        content_source: ContentSource,
        scores: dict,
    ) -> None:
        """Print summary of results."""
        print("\n" + "=" * 70)
        print("IdeaRank Summary")
        print("=" * 70)
        
        print(f"\nContent Source: {content_source.name}")
        print(f"Videos analyzed: {len(scores)}")
        
        # Sort by score
        sorted_scores = sorted(
            scores.items(),
            key=lambda x: x[1].score,
            reverse=True,
        )
        
        print(f"\nTop 5 videos by IdeaRank:")
        print("-" * 70)
        
        for i, (item_id, score) in enumerate(sorted_scores[:5], 1):
            item = next(item for item in content_source.content_items if item.id == item_id)
            print(f"\n{i}. {item.title[:60]}")
            print(f"   Score: {score.score:.4f} | Gates: {'✓' if score.passes_gates else '✗'}")
            print(f"   U={score.uniqueness.score:.3f} | "
                  f"C={score.cohesion.score:.3f} | "
                  f"L={score.learning.score:.3f} | "
                  f"Q={score.quality.score:.3f} | "
                  f"T={score.trust.score:.3f}")
        
        print("\n" + "=" * 70)

