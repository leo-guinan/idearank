"""
Medium Archive Pipeline

End-to-end pipeline for processing Medium blog archives and generating IdeaRank scores.
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
from pathlib import Path

from idearank.models import ContentItem, ContentSource
from idearank.pipeline import IdeaRankPipeline
from idearank.config import IdeaRankConfig
from idearank.integrations.storage import SQLiteStorage
from idearank.integrations.medium import MediumArchiveClient, MediumPost, MediumUser
from idearank.providers.embeddings import EmbeddingProvider
from idearank.providers.topics import TopicModelProvider
from idearank.providers.neighborhoods import NeighborhoodProvider

logger = logging.getLogger(__name__)


class MediumPipeline:
    """Pipeline for processing Medium archives through IdeaRank."""
    
    def __init__(
        self,
        storage: SQLiteStorage,
        embedding_provider: EmbeddingProvider,
        topic_provider: TopicModelProvider,
        neighborhood_provider: NeighborhoodProvider,
        config: Optional[IdeaRankConfig] = None,
        semantic_extractor = None,
    ):
        """
        Initialize Medium pipeline.
        
        Args:
            storage: SQLite storage for persistence
            embedding_provider: Provider for generating embeddings
            topic_provider: Provider for topic modeling
            neighborhood_provider: Provider for finding similar content
            config: Optional IdeaRank configuration
            semantic_extractor: Optional semantic extractor for content decomposition
        """
        self.storage = storage
        self.medium_client = MediumArchiveClient()
        
        # Create IdeaRank pipeline
        if config is None:
            config = IdeaRankConfig.default()
            
        self.pipeline = IdeaRankPipeline(
            config=config,
            embedding_provider=embedding_provider,
            topic_provider=topic_provider,
            neighborhood_provider=neighborhood_provider,
            storage=storage,  # Pass storage for chunk/semantic persistence
            semantic_extractor=semantic_extractor,  # For semantic decomposition
        )
        
    def process_archive(
        self,
        archive_path: str,
        limit: Optional[int] = None,
        skip_drafts: bool = True,
    ) -> tuple[ContentSource, Dict]:
        """
        Process a Medium archive ZIP file.
        
        Args:
            archive_path: Path to Medium export ZIP file
            limit: Optional limit on number of posts to process
            skip_drafts: Whether to skip draft posts
            
        Returns:
            Tuple of (content source, statistics dict)
        """
        logger.info("=" * 70)
        logger.info(f"Processing Medium archive: {archive_path}")
        logger.info("=" * 70)
        logger.info("")
        
        # Load archive
        logger.info("[1/6] Loading Medium archive...")
        user, posts = self.medium_client.load_archive(archive_path)
        
        # Track filtering stats
        original_count = len(posts)
        filtered_drafts = 0
        filtered_comments = 0
        
        # Filter drafts if needed
        if skip_drafts:
            before = len(posts)
            posts = [p for p in posts if not p.is_draft]
            filtered_drafts = before - len(posts)
        
        # Always filter comments (short responses)
        before = len(posts)
        posts = [p for p in posts if not p.is_comment]
        filtered_comments = before - len(posts)
        
        # Log filtering results
        if filtered_drafts > 0 or filtered_comments > 0:
            logger.info(f"Filtered out: {filtered_drafts} drafts, {filtered_comments} comments")
            
        # Filter to limit
        if limit:
            posts = posts[:limit]
            logger.info(f"Limited to {len(posts)} posts")
            
        logger.info(f"✓ Loaded {len(posts)} posts from {user.name} (out of {original_count} total)")
        logger.info("")
        
        # Convert to IdeaRank format
        logger.info("[2/6] Converting to IdeaRank format...")
        content_items, content_source = self._convert_to_idearank_format(user, posts)
        logger.info(f"✓ Converted {len(content_items)} posts")
        logger.info("")
        
        # Save to database
        logger.info("[3/6] Saving to SQLite database...")
        self.storage.save_content_source(content_source)
        for i, item in enumerate(content_items, 1):
            self.storage.save_content_item(item)
            if i % 10 == 0:
                logger.info(f"  Saved {i}/{len(content_items)} posts...")
        logger.info(f"✓ Saved {len(content_items)} posts to database")
        logger.info("")
        
        # Process through IdeaRank pipeline
        logger.info("[4/6] Generating embeddings...")
        self.pipeline.process_content_batch(content_items)
        logger.info(f"✓ Generated {len(content_items)} embeddings")
        logger.info("")
        
        logger.info("[5/6] Indexing embeddings...")
        self.pipeline.index_content(content_items)
        logger.info(f"✓ Indexed {len(content_items)} posts")
        logger.info("")
        
        # Compute IdeaRank scores
        logger.info("[6/6] Computing IdeaRank scores...")
        scores = []
        for i, item in enumerate(content_items, 1):
            logger.info(f"  Scoring post {i}/{len(content_items)}: {item.title[:50]}...")
            score = self.pipeline.score_content_item(item, content_source)
            scores.append(score)
            
            # Save score to database
            self.storage.save_content_score(item.id, content_source.id, score)
            
        logger.info(f"✓ Computed {len(scores)} IdeaRank scores")
        logger.info("")
        
        # Compute source-level score
        source_scorer = self.pipeline.source_scorer
        # Convert list of scores to dict keyed by content_item_id
        scores_dict = {score.content_item_id: score for score in scores}
        source_score = source_scorer.score_source(content_source, content_scores=scores_dict)
        self.storage.save_source_score(content_source.id, source_score)
        
        # Calculate statistics
        stats = self._calculate_statistics(content_source, content_items, scores)
        
        logger.info("=" * 70)
        logger.info("Pipeline complete!")
        logger.info("=" * 70)
        
        return content_source, stats
        
    def _convert_to_idearank_format(
        self,
        user: MediumUser,
        posts: List[MediumPost],
    ) -> tuple[List[ContentItem], ContentSource]:
        """Convert Medium posts to IdeaRank format."""
        content_items = []
        
        for post in posts:
            # Use published_at if available, otherwise use current time (timezone-aware)
            published_at = post.published_at or datetime.now(timezone.utc)
            
            # Ensure timezone-aware
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=timezone.utc)
            
            content_item = ContentItem(
                id=post.id,
                content_source_id=f"medium_{user.username}",
                title=post.title,
                description=post.content[:500],  # First 500 chars as description
                body=post.content,
                published_at=published_at,
                captured_at=datetime.now(timezone.utc),
                view_count=post.claps,  # Use claps as proxy for views
                impression_count=post.claps * 10,  # Rough estimate
                watch_time_seconds=float(post.word_count * 2),  # ~2 seconds per word
                avg_view_duration=float(post.word_count * 2),
                content_duration=float(post.word_count * 2),
                tags=post.tags,
                embedding=None,
                topic_mixture=None,
            )
            content_items.append(content_item)
            
        # Create content source
        # Find earliest publish date (timezone-aware)
        earliest_date = datetime.now(timezone.utc)
        if posts:
            dates = [p.published_at for p in posts if p.published_at]
            if dates:
                earliest_date = min(dates)
                if earliest_date.tzinfo is None:
                    earliest_date = earliest_date.replace(tzinfo=timezone.utc)
        
        content_source = ContentSource(
            id=f"medium_{user.username}",
            name=user.name,
            description=user.bio or f"Medium blog by {user.name}",
            created_at=earliest_date,
            subscriber_count=user.follower_count,
            total_views=sum(p.claps for p in posts),
            content_items=content_items,
        )
        
        return content_items, content_source
        
    def _calculate_statistics(
        self,
        content_source: ContentSource,
        content_items: List[ContentItem],
        scores: List,
    ) -> Dict:
        """Calculate summary statistics."""
        score_values = [s.score for s in scores]
        
        stats = {
            'total_posts': len(content_items),
            'avg_score': sum(score_values) / len(score_values) if score_values else 0,
            'min_score': min(score_values) if score_values else 0,
            'max_score': max(score_values) if score_values else 0,
            'total_claps': sum(item.view_count for item in content_items),
            'avg_word_count': sum(len(item.body.split()) for item in content_items) / len(content_items) if content_items else 0,
            'date_range': {
                'earliest': min(item.published_at for item in content_items) if content_items else None,
                'latest': max(item.published_at for item in content_items) if content_items else None,
            },
        }
        
        return stats


def process_medium_archive(
    archive_path: str,
    storage: SQLiteStorage,
    embedding_provider: EmbeddingProvider,
    topic_provider: TopicModelProvider,
    neighborhood_provider: NeighborhoodProvider,
    limit: Optional[int] = None,
    skip_drafts: bool = True,
    config: Optional[IdeaRankConfig] = None,
    semantic_extractor = None,
) -> tuple[ContentSource, Dict]:
    """
    Convenience function to process a Medium archive.
    
    Args:
        archive_path: Path to Medium export ZIP file
        storage: SQLite storage
        embedding_provider: Embedding provider
        topic_provider: Topic provider
        neighborhood_provider: Neighborhood provider
        limit: Optional limit on posts
        skip_drafts: Whether to skip drafts
        config: Optional IdeaRank config
        semantic_extractor: Optional semantic extractor for content decomposition
        
    Returns:
        Tuple of (content source, statistics)
    """
    pipeline = MediumPipeline(
        storage=storage,
        embedding_provider=embedding_provider,
        topic_provider=topic_provider,
        neighborhood_provider=neighborhood_provider,
        config=config,
        semantic_extractor=semantic_extractor,
    )
    
    return pipeline.process_archive(
        archive_path=archive_path,
        limit=limit,
        skip_drafts=skip_drafts,
    )

