"""Example showing how to customize IdeaRank weights and hyperparameters.

Demonstrates different configurations for different use cases.
"""

from datetime import datetime, timedelta

from idearank import IdeaRankConfig, Video, Channel
from idearank.config import FactorWeights
from idearank.pipeline import IdeaRankPipeline
from idearank.providers import (
    DummyEmbeddingProvider,
    DummyTopicModelProvider,
    DummyNeighborhoodProvider,
)


def create_test_channel() -> Channel:
    """Create a simple test channel."""
    videos = []
    base_time = datetime(2024, 1, 1)
    
    for i in range(5):
        video = Video(
            id=f"video_{i}",
            channel_id="test_channel",
            title=f"Video {i}",
            description=f"Content for video {i}",
            transcript=f"Transcript content {i} " * 100,
            published_at=base_time + timedelta(days=i * 30),
            snapshot_time=datetime.utcnow(),
            view_count=1000,
            impression_count=5000,
            watch_time_seconds=100000.0,
            avg_view_duration=200.0,
            video_duration=300.0,
            has_citations=True,
            citation_count=5,
            source_diversity_score=0.7,
        )
        videos.append(video)
    
    return Channel(
        id="test_channel",
        name="Test Channel",
        description="A test channel",
        created_at=datetime(2023, 1, 1),
        videos=videos,
    )


def compare_configurations():
    """Compare different weight configurations."""
    print("=" * 80)
    print("IdeaRank Configuration Comparison")
    print("=" * 80)
    
    # Create test data
    channel = create_test_channel()
    test_video = channel.videos[2]
    
    # Define different configurations
    configs = {
        "Default": IdeaRankConfig.default(),
        
        "Uniqueness-Focused": IdeaRankConfig(
            weights=FactorWeights(
                uniqueness=0.6,  # Emphasize originality
                cohesion=0.1,
                learning=0.2,
                quality=0.05,
                trust=0.05,
            )
        ),
        
        "Quality-Focused": IdeaRankConfig(
            weights=FactorWeights(
                uniqueness=0.15,
                cohesion=0.15,
                learning=0.15,
                quality=0.5,  # Emphasize engagement
                trust=0.05,
            )
        ),
        
        "Learning-Focused": IdeaRankConfig(
            weights=FactorWeights(
                uniqueness=0.2,
                cohesion=0.15,
                learning=0.5,  # Emphasize progression
                quality=0.1,
                trust=0.05,
            )
        ),
        
        "Trust-Focused": IdeaRankConfig(
            weights=FactorWeights(
                uniqueness=0.2,
                cohesion=0.2,
                learning=0.2,
                quality=0.1,
                trust=0.3,  # Emphasize verifiability
            )
        ),
    }
    
    # Score with each configuration
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n{config_name} Configuration")
        print("-" * 80)
        
        # Create pipeline
        pipeline = IdeaRankPipeline(
            config=config,
            embedding_provider=DummyEmbeddingProvider(seed=42),
            topic_provider=DummyTopicModelProvider(seed=42),
            neighborhood_provider=DummyNeighborhoodProvider(),
        )
        
        # Process and score
        pipeline.process_videos_batch(channel.videos)
        pipeline.index_videos(channel.videos)
        score = pipeline.score_video(test_video, channel)
        
        results[config_name] = score
        
        # Print results
        print(f"Overall Score: {score.score:.4f}")
        print(f"  U={score.uniqueness.score:.4f} (weight={config.weights.uniqueness})")
        print(f"  C={score.cohesion.score:.4f} (weight={config.weights.cohesion})")
        print(f"  L={score.learning.score:.4f} (weight={config.weights.learning})")
        print(f"  Q={score.quality.score:.4f} (weight={config.weights.quality})")
        print(f"  T={score.trust.score:.4f} (weight={config.weights.trust})")
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("Summary Comparison")
    print("=" * 80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1].score, reverse=True)
    
    for rank, (config_name, score) in enumerate(sorted_results, 1):
        print(f"{rank}. {config_name:20} | Score: {score.score:.4f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    compare_configurations()

