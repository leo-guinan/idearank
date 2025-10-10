"""Example showing how to customize IdeaRank weights and hyperparameters.

Demonstrates different configurations for different use cases.
"""

from datetime import datetime, timedelta

from idearank import IdeaRankConfig, ContentItem, ContentSource
from idearank.config import FactorWeights
from idearank.pipeline import IdeaRankPipeline
from idearank.providers import (
    DummyEmbeddingProvider,
    DummyTopicModelProvider,
    DummyNeighborhoodProvider,
)


def create_test_source() -> ContentSource:
    """Create a simple test content source."""
    content_items = []
    base_time = datetime(2024, 1, 1)
    
    for i in range(5):
        item = ContentItem(
            id=f"item_{i}",
            content_source_id="test_source",
            title=f"Content Item {i}",
            description=f"Content description {i}",
            body=f"Content body {i} " * 100,
            published_at=base_time + timedelta(days=i * 30),
            captured_at=datetime.utcnow(),
            view_count=1000,
            impression_count=5000,
            watch_time_seconds=100000.0,
            avg_view_duration=200.0,
            content_duration=300.0,
            has_citations=True,
            citation_count=5,
            source_diversity_score=0.7,
        )
        content_items.append(item)
    
    return ContentSource(
        id="test_source",
        name="Test Content Source",
        description="A test content source",
        created_at=datetime(2023, 1, 1),
        content_items=content_items,
    )


def compare_configurations():
    """Compare different weight configurations."""
    print("=" * 80)
    print("IdeaRank Configuration Comparison")
    print("=" * 80)
    
    # Create test data
    content_source = create_test_source()
    test_item = content_source.content_items[2]
    
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
        pipeline.process_content_batch(content_source.content_items)
        pipeline.index_content(content_source.content_items)
        score = pipeline.score_content_item(test_item, content_source)
        
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

