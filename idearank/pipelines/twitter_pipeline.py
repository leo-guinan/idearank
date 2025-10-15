"""Twitter content processing pipeline for IdeaRank.

Processes Twitter archives from community-archive.org into IdeaRank scores.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from idearank.models import ContentItem, ContentSource
from idearank.integrations.twitter import TwitterArchive, TwitterPost
from idearank.integrations.storage import SQLiteStorage
from idearank.providers.embeddings import EmbeddingProvider
from idearank.providers.topics import TopicModelProvider
from idearank.providers.chroma import ChromaProvider
from idearank.pipeline import IdeaRankPipeline

logger = logging.getLogger(__name__)


class TwitterPipeline:
    """Pipeline for processing Twitter archives into IdeaRank scores."""
    
    def __init__(
        self,
        storage: SQLiteStorage,
        embedding_provider: EmbeddingProvider,
        topic_provider: TopicModelProvider,
        chroma_provider: ChromaProvider,
        semantic_extractor = None,
    ):
        """Initialize Twitter pipeline.
        
        Args:
            storage: SQLite storage for persistence
            embedding_provider: For generating embeddings
            topic_provider: For topic modeling
            chroma_provider: For vector search
            semantic_extractor: Optional semantic extractor for content decomposition
        """
        self.storage = storage
        self.embedding_provider = embedding_provider
        self.topic_provider = topic_provider
        self.chroma_provider = chroma_provider
        
        # Create config for IdeaRank pipeline
        from idearank.config import IdeaRankConfig
        config = IdeaRankConfig()
        
        # Create main pipeline
        self.pipeline = IdeaRankPipeline(
            config=config,
            embedding_provider=embedding_provider,
            topic_provider=topic_provider,
            neighborhood_provider=chroma_provider,
            storage=storage,  # Pass storage for chunk/semantic persistence
            semantic_extractor=semantic_extractor,  # For semantic decomposition
        )
    
    def process_archive(
        self,
        archive: TwitterArchive,
        limit: Optional[int] = None,
        batch_size: int = 50,
    ) -> Dict[str, Any]:
        """Process a Twitter archive into IdeaRank scores.
        
        Args:
            archive: Twitter archive to process
            limit: Maximum number of posts to process
            batch_size: Number of posts to process in each batch
            
        Returns:
            Dict with processing results and statistics
        """
        logger.info(f"Processing Twitter archive for @{archive.username}")
        logger.info(f"  Total posts: {archive.total_posts}")
        logger.info(f"  Date range: {archive.date_range[0]} to {archive.date_range[1]}")
        
        # Limit posts if specified
        posts_to_process = archive.posts
        if limit:
            posts_to_process = archive.posts[:limit]
            logger.info(f"  Processing first {limit} posts")
        
        # Create or get content source
        content_source = self._get_or_create_source(archive)
        
        # Convert posts to ContentItem objects
        content_items = self._convert_posts_to_content(posts_to_process, content_source)
        
        logger.info(f"Converted {len(content_items)} posts to content items")
        
        # Save content items to database first
        logger.info(f"\n[3/6] Saving to SQLite database...")
        for i, item in enumerate(content_items, 1):
            self.storage.save_content_item(item)
            if i % 10 == 0:
                logger.info(f"  Saved {i}/{len(content_items)} posts...")
        logger.info(f"✓ Saved {len(content_items)} posts to database")
        
        # Process in batches
        processed_items = []
        for i in range(0, len(content_items), batch_size):
            batch = content_items[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(content_items) + batch_size - 1)//batch_size}")
            
            # Process batch through IdeaRank pipeline
            batch_results = self.pipeline.process_content_batch(batch)
            processed_items.extend(batch_results)
        
        # Score the processed content items
        results = []
        for item in processed_items:
            score_result = self.pipeline.score_content_item(item, content_source)
            results.append(score_result)
        
        # Calculate statistics
        stats = self._calculate_statistics(results, archive)
        
        logger.info(f"✓ Completed processing {len(results)} posts")
        logger.info(f"  Average score: {stats['average_score']:.4f}")
        logger.info(f"  Score range: {stats['min_score']:.4f} - {stats['max_score']:.4f}")
        
        return {
            'processed_count': len(results),
            'content_source_id': content_source.id,
            'username': archive.username,
            'statistics': stats,
            'results': results,
        }
    
    def _get_or_create_source(self, archive: TwitterArchive) -> ContentSource:
        """Get or create a content source for the Twitter user.
        
        Args:
            archive: Twitter archive
            
        Returns:
            ContentSource object
        """
        # Create new content source
        source_id = f"twitter_{archive.username}"
        content_source = ContentSource(
            id=source_id,
            name=f"@{archive.username} (Twitter)",
            description=f"Twitter archive from community-archive.org",
            created_at=archive.date_range[0],  # Use earliest post date
            subscriber_count=0,  # Not available from archive
            total_views=0,  # Not available from archive
        )
        
        # Save source (this handles both create and update)
        self.storage.save_content_source(content_source)
        logger.info(f"Created/updated content source: {source_id}")
        
        return content_source
    
    def _convert_posts_to_content(
        self,
        posts: List[TwitterPost],
        content_source: ContentSource,
    ) -> List[ContentItem]:
        """Convert Twitter posts to ContentItem objects for IdeaRank processing.
        
        Args:
            posts: List of Twitter posts
            content_source: ContentSource object
            
        Returns:
            List of ContentItem objects
        """
        content_items = []
        
        for post in posts:
            # Create content item ID
            item_id = f"twitter_{post.id}"
            
            # Calculate engagement metrics
            total_engagement = post.retweet_count + post.favorite_count + post.reply_count
            
            # Estimate watch time (Twitter doesn't have this, so use engagement as proxy)
            # Assume average "reading time" of 30 seconds for tweets
            estimated_watch_time = 30 * max(1, total_engagement // 10)  # Scale with engagement
            
            # Create ContentItem object
            content_item = ContentItem(
                id=item_id,
                content_source_id=content_source.id,
                title=post.text[:100] + "..." if len(post.text) > 100 else post.text,
                description=post.text,
                body=post.text,  # Full tweet text
                published_at=post.created_at,
                captured_at=datetime.now(),
                
                # Analytics (using Twitter engagement as proxy)
                view_count=total_engagement,  # Use total engagement as proxy
                impression_count=max(total_engagement, 1),  # At least 1 impression
                watch_time_seconds=estimated_watch_time,
                avg_view_duration=estimated_watch_time,
                content_duration=30.0,  # Estimated reading time
                
                # Trust signals
                has_citations=len(post.hashtags) > 0 or len(post.mentions) > 0,
                citation_count=len(post.hashtags) + len(post.mentions),
                source_diversity_score=min(1.0, len(post.hashtags) * 0.1 + len(post.mentions) * 0.05),
                correction_count=0,  # Assume no corrections for archived posts
                
                # Tags from hashtags
                tags=post.hashtags,
            )
            
            content_items.append(content_item)
        
        return content_items
    
    def _calculate_statistics(
        self,
        results: List[Dict[str, Any]],
        archive: TwitterArchive,
    ) -> Dict[str, Any]:
        """Calculate processing statistics.
        
        Args:
            results: List of scoring results
            archive: Original Twitter archive
            
        Returns:
            Dict with statistics
        """
        if not results:
            return {
                'average_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'score_std': 0.0,
                'total_engagement': 0,
                'average_engagement': 0.0,
            }
        
        # Extract scores
        scores = [result.score for result in results]
        
        # Calculate engagement
        total_engagement = sum(
            post.retweet_count + post.favorite_count + post.reply_count
            for post in archive.posts
        )
        
        return {
            'average_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'score_std': (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5,
            'total_engagement': total_engagement,
            'average_engagement': total_engagement / len(archive.posts) if archive.posts else 0,
            'posts_with_hashtags': sum(1 for post in archive.posts if post.hashtags),
            'posts_with_mentions': sum(1 for post in archive.posts if post.mentions),
            'unique_hashtags': len(set(tag for post in archive.posts for tag in post.hashtags)),
            'unique_mentions': len(set(mention for post in archive.posts for mention in post.mentions)),
        }


def process_twitter_archive(
    archive: TwitterArchive,
    storage: SQLiteStorage,
    embedding_provider: EmbeddingProvider,
    topic_provider: TopicModelProvider,
    chroma_provider: ChromaProvider,
    limit: Optional[int] = None,
    semantic_extractor = None,
) -> Dict[str, Any]:
    """Convenience function to process a Twitter archive.
    
    Args:
        archive: Twitter archive object
        storage: SQLite storage
        embedding_provider: Embedding provider
        topic_provider: Topic provider
        chroma_provider: Chroma provider
        limit: Maximum posts to process
        semantic_extractor: Optional semantic extractor for content decomposition
        
    Returns:
        Processing results
    """
    # Create pipeline and process
    pipeline = TwitterPipeline(
        storage=storage,
        embedding_provider=embedding_provider,
        topic_provider=topic_provider,
        chroma_provider=chroma_provider,
        semantic_extractor=semantic_extractor,
    )
    
    results = pipeline.process_archive(archive, limit)
    results['success'] = True
    
    return results
