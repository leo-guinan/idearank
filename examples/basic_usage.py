"""Basic usage example for IdeaRank.

Demonstrates:
1. Creating videos and channels
2. Setting up the pipeline
3. Computing video and channel scores
4. Using the network layer
"""

from datetime import datetime, timedelta
import numpy as np

from idearank import IdeaRankConfig, Video, Channel
from idearank.pipeline import IdeaRankPipeline
from idearank.providers import (
    DummyEmbeddingProvider,
    DummyTopicModelProvider,
    DummyNeighborhoodProvider,
)


def create_sample_videos(channel_id: str, count: int = 5) -> list[Video]:
    """Create sample videos for testing."""
    videos = []
    base_time = datetime(2024, 1, 1)
    
    for i in range(count):
        video = Video(
            id=f"{channel_id}_video_{i}",
            channel_id=channel_id,
            title=f"Video {i}: Introduction to Topic {i % 3}",
            description=f"This is a detailed video about topic {i % 3}. " * 10,
            transcript=f"In this video we explore topic {i % 3} in depth. " * 50,
            published_at=base_time + timedelta(days=i * 7),
            snapshot_time=datetime.utcnow(),
            # Analytics
            view_count=1000 * (i + 1),
            impression_count=5000 * (i + 1),
            watch_time_seconds=float(1000 * (i + 1) * 120),
            avg_view_duration=120.0 + i * 10,
            video_duration=300.0,
            # Trust signals
            has_citations=i % 2 == 0,
            citation_count=i * 2,
            source_diversity_score=0.5 + (i * 0.1),
        )
        videos.append(video)
    
    return videos


def main():
    """Run the basic example."""
    print("=" * 60)
    print("IdeaRank Basic Usage Example")
    print("=" * 60)
    
    # 1. Create configuration
    print("\n1. Setting up configuration...")
    config = IdeaRankConfig.default()
    config.network.enabled = True  # Enable KnowledgeRank
    print(f"   Weights: U={config.weights.uniqueness}, C={config.weights.cohesion}, "
          f"L={config.weights.learning}, Q={config.weights.quality}, T={config.weights.trust}")
    
    # 2. Initialize providers (using dummy providers for demo)
    print("\n2. Initializing providers...")
    embedding_provider = DummyEmbeddingProvider(dimension=384, seed=42)
    topic_provider = DummyTopicModelProvider(num_topics=50, seed=42)
    neighborhood_provider = DummyNeighborhoodProvider()
    
    # 3. Create pipeline
    print("\n3. Creating IdeaRank pipeline...")
    pipeline = IdeaRankPipeline(
        config=config,
        embedding_provider=embedding_provider,
        topic_provider=topic_provider,
        neighborhood_provider=neighborhood_provider,
    )
    
    # 4. Create sample data
    print("\n4. Creating sample channels and videos...")
    channels = []
    
    for i in range(3):
        channel = Channel(
            id=f"channel_{i}",
            name=f"Channel {i}",
            description=f"A channel about various topics",
            created_at=datetime(2023, 1, 1),
            videos=create_sample_videos(f"channel_{i}", count=10),
        )
        channels.append(channel)
    
    total_videos = sum(len(c.videos) for c in channels)
    print(f"   Created {len(channels)} channels with {total_videos} total videos")
    
    # 5. Score a single video
    print("\n5. Scoring a single video...")
    test_video = channels[0].videos[3]
    test_channel = channels[0]
    
    # Process and index (needed for neighborhood search)
    pipeline.process_videos_batch(test_channel.videos)
    pipeline.index_videos(test_channel.videos)
    
    video_score = pipeline.score_video(test_video, test_channel)
    
    print(f"\n   Video: {test_video.title}")
    print(f"   Overall IdeaRank Score: {video_score.score:.4f}")
    print(f"   Factor Breakdown:")
    print(f"     - Uniqueness (U):  {video_score.uniqueness.score:.4f}")
    print(f"     - Cohesion (C):    {video_score.cohesion.score:.4f}")
    print(f"     - Learning (L):    {video_score.learning.score:.4f}")
    print(f"     - Quality (Q):     {video_score.quality.score:.4f}")
    print(f"     - Trust (T):       {video_score.trust.score:.4f}")
    print(f"   Passes Gates: {video_score.passes_gates}")
    
    # 6. Score a channel
    print("\n6. Scoring a channel...")
    channel_score = pipeline.score_channel(test_channel)
    
    print(f"\n   Channel: {test_channel.name}")
    print(f"   Channel IdeaRank Score: {channel_score.score:.4f}")
    print(f"   Mean Video Score: {channel_score.mean_video_score:.4f}")
    print(f"   AUL Bonus: {channel_score.aul_bonus:.4f}")
    print(f"   Videos in Window: {channel_score.video_count}")
    print(f"   Crystallization Detected: {channel_score.crystallization_detected}")
    
    # 7. Compute network scores
    print("\n7. Computing KnowledgeRank across channels...")
    kr_scores = pipeline.score_channels_with_network(channels)
    
    print("\n   Channel Rankings:")
    print("   " + "-" * 50)
    
    # Sort by KR score
    sorted_channels = sorted(
        kr_scores.items(),
        key=lambda x: x[1].knowledge_rank,
        reverse=True
    )
    
    for channel_id, kr_score in sorted_channels:
        channel_name = next(c.name for c in channels if c.id == channel_id)
        print(f"   {channel_name:15} | KR: {kr_score.knowledge_rank:.4f} | "
              f"IR: {kr_score.idea_rank:.4f} | "
              f"Influence: {kr_score.influence_bonus:+.4f}")
        print(f"                   | Out edges: {len(kr_score.outgoing_influence)} | "
              f"In edges: {len(kr_score.incoming_influence)}")
    
    # 8. Export results
    print("\n8. Exporting results to dict...")
    video_dict = video_score.to_dict()
    channel_dict = channel_score.to_dict()
    kr_dict = {cid: score.to_dict() for cid, score in kr_scores.items()}
    
    print(f"   Exported {len(video_dict)} video fields")
    print(f"   Exported {len(channel_dict)} channel fields")
    print(f"   Exported {len(kr_dict)} KnowledgeRank scores")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

