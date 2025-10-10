"""Simple test of YouTube pipeline with mock data.

This shows the pipeline structure without requiring YouTube API access.
"""

import logging
from datetime import datetime, timedelta

from idearank import IdeaRankConfig, ContentItem, ContentSource
from idearank.pipeline import IdeaRankPipeline
from idearank.providers import DummyTopicModelProvider
from idearank.providers import DummyEmbeddingProvider, DummyNeighborhoodProvider
from idearank.integrations.storage import SQLiteStorage
from idearank.integrations.youtube import YouTubeVideoData

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mock_youtube_data() -> list[YouTubeVideoData]:
    """Create 3 mock YouTube videos for testing."""
    base_time = datetime(2024, 1, 1)
    
    videos = [
        YouTubeVideoData(
            video_id="mock_video_1",
            channel_id="UCmock_channel",
            title="Introduction to Supply Chain Management",
            description="Learn the basics of modern supply chain operations and logistics.",
            published_at=base_time,
            view_count=15000,
            like_count=450,
            comment_count=89,
            duration_seconds=900,  # 15 minutes
            transcript="Welcome to our supply chain series. Today we'll cover the fundamentals of supply chain management, including procurement, inventory management, and distribution strategies. Let's dive into the key concepts that drive modern logistics operations.",
            transcript_source="youtube",
            tags=["supply chain", "logistics", "business"],
        ),
        YouTubeVideoData(
            video_id="mock_video_2",
            channel_id="UCmock_channel",
            title="Digital Transformation in Manufacturing",
            description="How Industry 4.0 technologies are reshaping manufacturing processes.",
            published_at=base_time + timedelta(days=7),
            view_count=12000,
            like_count=380,
            comment_count=72,
            duration_seconds=1200,  # 20 minutes
            transcript="Digital transformation is revolutionizing manufacturing. We're seeing the rise of smart factories, IoT sensors, and AI-driven quality control. These technologies enable real-time monitoring and predictive maintenance, reducing downtime and improving efficiency across the production line.",
            transcript_source="youtube",
            tags=["manufacturing", "digital transformation", "industry 4.0"],
        ),
        YouTubeVideoData(
            video_id="mock_video_3",
            channel_id="UCmock_channel",
            title="Sustainable Supply Chains for the Future",
            description="Exploring eco-friendly practices and circular economy principles in logistics.",
            published_at=base_time + timedelta(days=14),
            view_count=18000,
            like_count=520,
            comment_count=105,
            duration_seconds=1080,  # 18 minutes
            transcript="Sustainability is no longer optional in supply chain management. We're exploring circular economy principles, carbon-neutral logistics, and ethical sourcing. Companies that embrace these practices are seeing both environmental and economic benefits through waste reduction and improved brand reputation.",
            transcript_source="youtube",
            tags=["sustainability", "supply chain", "circular economy"],
        ),
    ]
    
    return videos


def convert_to_idearank_content(yt_videos: list[YouTubeVideoData]) -> tuple[list[ContentItem], ContentSource]:
    """Convert mock YouTube data to IdeaRank format."""
    content_items = []
    
    for yt_video in yt_videos:
        # Estimate analytics
        impression_count = yt_video.view_count * 5
        avg_duration_estimate = yt_video.duration_seconds * 0.55  # 55% completion
        watch_time = yt_video.view_count * avg_duration_estimate
        
        item = ContentItem(
            id=yt_video.video_id,
            content_source_id=yt_video.channel_id,
            title=yt_video.title,
            description=yt_video.description,
            body=yt_video.transcript or "",
            published_at=yt_video.published_at,
            captured_at=datetime.now(),
            view_count=yt_video.view_count,
            impression_count=impression_count,
            watch_time_seconds=float(watch_time),
            avg_view_duration=float(avg_duration_estimate),
            content_duration=float(yt_video.duration_seconds),
            has_citations=False,
            citation_count=0,
            source_diversity_score=0.5,
            correction_count=0,
            tags=yt_video.tags or [],
        )
        content_items.append(item)
    
    content_source = ContentSource(
        id="UCmock_channel",
        name="Mock Supply Chain Channel",
        description="Educational content about supply chain and logistics",
        created_at=min(item.published_at for item in content_items),
        content_items=content_items,
    )
    
    return content_items, content_source


def main():
    """Run simple pipeline test."""
    
    print("=" * 80)
    print("YouTube → IdeaRank Pipeline Test (Mock Data)")
    print("=" * 80)
    
    # 1. Create mock data
    print("\n[1/5] Creating mock YouTube videos...")
    yt_videos = create_mock_youtube_data()
    print(f"✓ Created {len(yt_videos)} mock videos")
    
    # 2. Convert to IdeaRank format
    print("\n[2/5] Converting to IdeaRank format...")
    content_items, content_source = convert_to_idearank_content(yt_videos)
    print(f"✓ Converted {len(content_items)} videos")
    
    # 3. Initialize providers (using Dummy for this test)
    print("\n[3/5] Setting up providers...")
    embedding_provider = DummyEmbeddingProvider(dimension=384, seed=42)
    neighborhood_provider = DummyNeighborhoodProvider()
    print("✓ Providers ready")
    
    # 4. Create pipeline
    print("\n[4/5] Creating IdeaRank pipeline...")
    config = IdeaRankConfig.default()
    config.uniqueness.k_global = 2  # Only 3 videos, so small k
    config.cohesion.k_intra = 2
    
    pipeline = IdeaRankPipeline(
        config=config,
        embedding_provider=embedding_provider,
        topic_provider=DummyTopicModelProvider(),
        neighborhood_provider=neighborhood_provider,
    )
    print("✓ Pipeline ready")
    
    # 5. Process content
    print("\n[5/5] Processing pipeline...")
    
    # Generate embeddings
    print("  - Generating embeddings...")
    pipeline.process_content_batch(content_items)
    print("  ✓ Embeddings generated")
    
    # Index in Chroma
    print("  - Indexing content...")
    pipeline.index_content(content_items)
    print("  ✓ Content indexed")
    
    # Score content
    print("  - Computing IdeaRank scores...")
    scores = {}
    for i, item in enumerate(content_items, 1):
        print(f"    Scoring {i}/{len(content_items)}: {item.title[:40]}...")
        score = pipeline.score_content_item(item, content_source)
        scores[item.id] = score
    print("  ✓ Scores computed")
    
    # 6. Save to SQLite
    print("\n[6/6] Saving to SQLite...")
    storage = SQLiteStorage(db_path="youtube_test.db")
    storage.save_content_source(content_source)
    for item, yt_data in zip(content_items, yt_videos):
        storage.save_content_item(item, yt_data)
    for item_id, score in scores.items():
        storage.save_content_score(item_id, content_source.id, score)
    storage.close()
    print("✓ Saved to youtube_test.db")
    
    # 7. Display results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1].score, reverse=True)
    
    for i, (item_id, score) in enumerate(sorted_scores, 1):
        item = next(item for item in content_items if item.id == item_id)
        print(f"\n{i}. {item.title}")
        print(f"   IdeaRank: {score.score:.4f} | Gates: {'✓' if score.passes_gates else '✗'}")
        print(f"   U={score.uniqueness.score:.3f} | "
              f"C={score.cohesion.score:.3f} | "
              f"L={score.learning.score:.3f} | "
              f"Q={score.quality.score:.3f} | "
              f"T={score.trust.score:.3f}")
    
    print("\n" + "=" * 80)
    print("Test complete! ✅")
    print("=" * 80)
    print("\nPipeline components verified:")
    print("  ✓ YouTube data conversion")
    print("  ✓ Embedding generation")
    print("  ✓ Vector indexing")
    print("  ✓ IdeaRank scoring (all 5 factors)")
    print("  ✓ SQLite storage")
    print("\nNext steps:")
    print("  - Run with real Chroma Cloud (update credentials in youtube_pipeline_demo.py)")
    print("  - Get YouTube API key and run on real channels")
    print("  - Scale up to more videos")


if __name__ == "__main__":
    main()

