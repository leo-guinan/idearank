"""End-to-end pipeline: Ghost Blog → IdeaRank scores.

Fetches Ghost blog posts, embeds, indexes, and scores.
"""

from typing import List, Optional
from datetime import datetime
import logging

from idearank.models import ContentItem, ContentSource
from idearank.config import IdeaRankConfig
from idearank.pipeline import IdeaRankPipeline
from idearank.integrations.ghost import GhostClient, GhostPostData
from idearank.integrations.storage import SQLiteStorage

logger = logging.getLogger(__name__)


class GhostPipeline:
    """Complete pipeline from Ghost blog URL to IdeaRank scores."""
    
    def __init__(
        self,
        idearank_pipeline: IdeaRankPipeline,
        ghost_client: GhostClient,
        storage: SQLiteStorage,
        config: Optional[IdeaRankConfig] = None,
    ):
        """Initialize Ghost pipeline.
        
        Args:
            idearank_pipeline: Configured IdeaRankPipeline with providers
            ghost_client: GhostClient for data fetching
            storage: SQLiteStorage for persistence
            config: IdeaRankConfig (if not already in idearank_pipeline)
        """
        self.idearank_pipeline = idearank_pipeline
        self.ghost_client = ghost_client
        self.storage = storage
        self.config = config or IdeaRankConfig.default()
    
    def process_blog(
        self,
        blog_url: str,
        max_posts: Optional[int] = 50,
        filter_query: Optional[str] = None,
    ) -> tuple[ContentSource, dict]:
        """Process entire Ghost blog.
        
        Args:
            blog_url: Ghost blog URL (e.g., "https://blog.example.com")
            max_posts: Maximum number of posts to process (None = all posts)
            filter_query: Optional Ghost filter (e.g., "tag:python")
            
        Returns:
            (ContentSource object, scores_dict)
        """
        logger.info(f"=" * 70)
        logger.info(f"Processing Ghost blog: {blog_url}")
        logger.info(f"=" * 70)
        
        # Step 1: Fetch Ghost posts
        logger.info("\n[1/6] Fetching Ghost posts...")
        ghost_posts = self.ghost_client.get_posts(
            limit=max_posts,
            filter_query=filter_query,
        )
        logger.info(f"✓ Fetched {len(ghost_posts)} posts")
        
        # Step 2: Get blog info
        blog_info = self.ghost_client.get_blog_info()
        
        # Step 3: Convert to IdeaRank ContentItem objects
        logger.info("\n[2/6] Converting to IdeaRank format...")
        content_items, content_source = self._convert_to_idearank_format(ghost_posts, blog_info)
        logger.info(f"✓ Converted {len(content_items)} posts")
        
        # Step 4: Save to SQLite
        logger.info("\n[3/6] Saving to SQLite database...")
        self.storage.save_content_source(content_source)
        for i, (item, ghost_data) in enumerate(zip(content_items, ghost_posts), 1):
            self.storage.save_content_item(item, ghost_data)
            if i % 10 == 0:
                logger.info(f"  Saved {i}/{len(content_items)} posts...")
        logger.info(f"✓ Saved {len(content_items)} posts to database")
        
        # Step 5: Generate embeddings
        logger.info("\n[4/6] Generating embeddings...")
        self.idearank_pipeline.process_content_batch(content_items)
        logger.info(f"✓ Generated {len(content_items)} embeddings")
        
        # Step 6: Index in Chroma
        logger.info("\n[5/6] Indexing embeddings in Chroma...")
        self.idearank_pipeline.index_content(content_items)
        logger.info(f"✓ Indexed {len(content_items)} posts in Chroma")
        
        # Step 7: Compute IdeaRank scores
        logger.info("\n[6/6] Computing IdeaRank scores...")
        scores = self._score_all_posts(content_items, content_source)
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
        logger.info(f"✓ Blog score: {source_score.score:.4f}")
        
        logger.info(f"\n{'=' * 70}")
        logger.info("Pipeline complete!")
        logger.info(f"{'=' * 70}")
        
        return content_source, scores
    
    def _convert_to_idearank_format(
        self,
        ghost_posts: List[GhostPostData],
        blog_info: dict,
    ) -> tuple[List[ContentItem], ContentSource]:
        """Convert Ghost posts to IdeaRank ContentItem and ContentSource objects."""
        content_items = []
        
        if not ghost_posts:
            raise ValueError("No posts provided")
        
        # Use blog URL as source ID
        blog_url = ghost_posts[0].blog_url
        
        for post in ghost_posts:
            # Estimate engagement metrics
            # For blogs, we'll use reading time and word count as proxies
            # Assume average post gets 100 views per 1000 words
            estimated_views = max(100, post.word_count // 10)
            impression_count = estimated_views * 5  # Similar to YouTube CTR
            
            # Watch time = views * reading time (in seconds)
            watch_time = estimated_views * (post.reading_time * 60)
            avg_duration = post.reading_time * 60  # Reading time in seconds
            
            # Create ContentItem object
            content_item = ContentItem(
                id=post.post_id,
                content_source_id=blog_url,  # Use blog URL as source ID
                title=post.title,
                description=post.meta_description or post.excerpt or '',
                body=post.plaintext,  # Use plaintext content as body
                published_at=post.published_at,
                captured_at=datetime.now(),
                # Analytics (estimated)
                view_count=estimated_views,
                impression_count=impression_count,
                watch_time_seconds=float(watch_time),
                avg_view_duration=float(avg_duration),
                content_duration=float(post.reading_time * 60),
                # Trust signals
                has_citations=False,  # Could detect citations in content
                citation_count=0,
                source_diversity_score=0.5,  # Neutral default
                correction_count=0,
                # Tags
                tags=post.tags,
            )
            content_items.append(content_item)
        
        # Create ContentSource object
        content_source = ContentSource(
            id=blog_url,
            name=blog_info.get('title', blog_url),
            description=blog_info.get('description', ''),
            created_at=min(item.published_at for item in content_items),
            content_items=content_items,
        )
        
        return content_items, content_source
    
    def _score_all_posts(
        self,
        content_items: List[ContentItem],
        content_source: ContentSource,
    ) -> dict:
        """Score all posts and return dict of {post_id: IdeaRankScore}."""
        scores = {}
        
        for i, item in enumerate(content_items, 1):
            logger.info(f"  Scoring post {i}/{len(content_items)}: {item.title[:50]}...")
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
        print("IdeaRank Summary - Ghost Blog")
        print("=" * 70)
        
        print(f"\nBlog: {content_source.name}")
        print(f"Posts analyzed: {len(scores)}")
        
        # Sort by score
        sorted_scores = sorted(
            scores.items(),
            key=lambda x: x[1].score,
            reverse=True,
        )
        
        print(f"\nTop 5 posts by IdeaRank:")
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

