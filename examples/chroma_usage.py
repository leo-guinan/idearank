"""Example using Chroma Cloud for embeddings and vector search.

This example shows how to:
1. Initialize ChromaProvider with your credentials
2. Use it for both embeddings and neighborhood search
3. Score videos using Chroma backend
"""

from datetime import datetime, timedelta
import os

from idearank import IdeaRankConfig, Video, Channel
from idearank.pipeline import IdeaRankPipeline
from idearank.providers import DummyTopicModelProvider

# Check if Chroma is available
try:
    from idearank.providers.chroma import ChromaProvider
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("ChromaDB not installed. Install with: pip install chromadb")
    print("Or: pip install -e '.[chroma]'")
    exit(1)


def create_sample_videos(channel_id: str, count: int = 10) -> list[Video]:
    """Create sample videos for testing."""
    videos = []
    base_time = datetime(2024, 1, 1)
    
    topics = [
        "machine learning fundamentals",
        "neural networks architecture",
        "deep learning optimization",
        "computer vision techniques",
        "natural language processing",
        "reinforcement learning",
        "transfer learning",
        "model deployment",
        "ML system design",
        "AI ethics and safety",
    ]
    
    for i in range(count):
        topic = topics[i % len(topics)]
        video = Video(
            id=f"{channel_id}_video_{i}",
            channel_id=channel_id,
            title=f"Video {i}: {topic.title()}",
            description=f"A comprehensive guide to {topic}. " * 5,
            transcript=f"In this video, we explore {topic} in depth. " * 20,
            published_at=base_time + timedelta(days=i * 14),
            snapshot_time=datetime.now(),
            # Analytics
            view_count=1000 * (i + 1),
            impression_count=5000 * (i + 1),
            watch_time_seconds=float(800 * (i + 1) * 180),
            avg_view_duration=180.0 + i * 20,
            video_duration=600.0,
            # Trust signals
            has_citations=i % 2 == 0,
            citation_count=i * 3,
            source_diversity_score=0.6 + (i * 0.04),
        )
        videos.append(video)
    
    return videos


def main():
    """Run the Chroma Cloud example."""
    print("=" * 70)
    print("IdeaRank with Chroma Cloud Example")
    print("=" * 70)
    
    # Load credentials from environment or use provided values
    api_key = os.getenv("CHROMA_API_KEY", "ck-BojTG2QscadMvcrtFX9cPrmbUKHwGJ9VKYrvq1Noa5LG")
    tenant = os.getenv("CHROMA_TENANT", "e59b3318-066b-4aa2-886a-c21fd8f81ef0")
    database = os.getenv("CHROMA_DATABASE", "Idea Nexus Ventures")
    
    print(f"\n1. Connecting to Chroma Cloud...")
    print(f"   Tenant: {tenant}")
    print(f"   Database: {database}")
    
    # Initialize Chroma provider (handles both embedding and search)
    try:
        chroma = ChromaProvider(
            api_key=api_key,
            tenant=tenant,
            database=database,
            embedding_function="default",  # Options: "default", "openai", "sentence-transformers"
            collection_name="idearank_demo",
        )
        print("   ✓ Connected successfully!")
    except Exception as e:
        print(f"   ✗ Failed to connect: {e}")
        print("\nMake sure your Chroma Cloud credentials are correct.")
        return
    
    # Create IdeaRank configuration
    print("\n2. Setting up IdeaRank configuration...")
    config = IdeaRankConfig.default()
    config.uniqueness.k_global = 20  # Fewer neighbors for demo
    config.cohesion.k_intra = 5
    
    # Create pipeline
    print("\n3. Creating IdeaRank pipeline with Chroma backend...")
    pipeline = IdeaRankPipeline(
        config=config,
        embedding_provider=chroma.get_embedding_provider(),
        topic_provider=DummyTopicModelProvider(),  # Still using dummy for topics
        neighborhood_provider=chroma.get_neighborhood_provider(),
    )
    
    # Create sample data
    print("\n4. Creating sample channels and videos...")
    channels = []
    
    for i in range(2):
        channel = Channel(
            id=f"ml_channel_{i}",
            name=f"ML Education Channel {i}",
            description=f"Educational content about machine learning",
            created_at=datetime(2023, 1, 1),
            videos=create_sample_videos(f"ml_channel_{i}", count=10),
        )
        channels.append(channel)
    
    total_videos = sum(len(c.videos) for c in channels)
    print(f"   Created {len(channels)} channels with {total_videos} videos")
    
    # Process videos (generate embeddings)
    print("\n5. Generating embeddings with Chroma...")
    all_videos = [v for c in channels for v in c.videos]
    pipeline.process_videos_batch(all_videos)
    print(f"   ✓ Generated {total_videos} embeddings")
    
    # Index videos in Chroma
    print("\n6. Indexing videos in Chroma Cloud...")
    pipeline.index_videos(all_videos)
    print(f"   ✓ Indexed {total_videos} videos")
    
    # Score a single video
    print("\n7. Scoring a video with Chroma-powered search...")
    test_video = channels[0].videos[5]
    test_channel = channels[0]
    
    video_score = pipeline.score_video(test_video, test_channel)
    
    print(f"\n   Video: {test_video.title}")
    print(f"   Overall IdeaRank Score: {video_score.score:.4f}")
    print(f"\n   Factor Breakdown:")
    print(f"     - Uniqueness (U):  {video_score.uniqueness.score:.4f}")
    print(f"       → Mean similarity to global corpus: "
          f"{video_score.uniqueness.components['mean_similarity']:.4f}")
    print(f"     - Cohesion (C):    {video_score.cohesion.score:.4f}")
    print(f"     - Learning (L):    {video_score.learning.score:.4f}")
    print(f"     - Quality (Q):     {video_score.quality.score:.4f}")
    print(f"     - Trust (T):       {video_score.trust.score:.4f}")
    
    # Score all channels
    print("\n8. Scoring channels...")
    for channel in channels:
        channel_score = pipeline.score_channel(channel)
        print(f"\n   {channel.name}:")
        print(f"     Channel Score: {channel_score.score:.4f}")
        print(f"     Mean Video Score: {channel_score.mean_video_score:.4f}")
        print(f"     AUL Bonus: {channel_score.aul_bonus:.4f}")
        print(f"     Crystallization: {channel_score.crystallization_detected}")
    
    # Compute network scores
    print("\n9. Computing KnowledgeRank across channels...")
    config.network.enabled = True
    kr_scores = pipeline.score_channels_with_network(channels)
    
    print("\n   Channel Rankings:")
    print("   " + "-" * 60)
    
    sorted_channels = sorted(
        kr_scores.items(),
        key=lambda x: x[1].knowledge_rank,
        reverse=True
    )
    
    for channel_id, kr_score in sorted_channels:
        channel_name = next(c.name for c in channels if c.id == channel_id)
        print(f"   {channel_name:25} | KR: {kr_score.knowledge_rank:.4f} | "
              f"IR: {kr_score.idea_rank:.4f}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print("\nNote: Your embeddings are now stored in Chroma Cloud and can be")
    print("      queried across sessions. Clear with collection.delete() if needed.")


if __name__ == "__main__":
    main()

