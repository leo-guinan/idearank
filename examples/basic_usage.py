"""Basic usage example for IdeaRank.

Demonstrates:
1. Creating content items and sources
2. Setting up the pipeline
3. Computing content and source scores
4. Using the network layer
"""

from datetime import datetime, timedelta
import numpy as np

from idearank import IdeaRankConfig, ContentItem, ContentSource
from idearank.pipeline import IdeaRankPipeline
from idearank.providers import (
    DummyEmbeddingProvider,
    DummyTopicModelProvider,
    DummyNeighborhoodProvider,
)


def create_sample_content(source_id: str, count: int = 5) -> list[ContentItem]:
    """Create sample content items for testing."""
    content_items = []
    base_time = datetime(2024, 1, 1)
    
    for i in range(count):
        item = ContentItem(
            id=f"{source_id}_item_{i}",
            content_source_id=source_id,
            title=f"Content {i}: Introduction to Topic {i % 3}",
            description=f"This is detailed content about topic {i % 3}. " * 10,
            body=f"In this content we explore topic {i % 3} in depth. " * 50,
            published_at=base_time + timedelta(days=i * 7),
            captured_at=datetime.utcnow(),
            # Analytics
            view_count=1000 * (i + 1),
            impression_count=5000 * (i + 1),
            watch_time_seconds=float(1000 * (i + 1) * 120),
            avg_view_duration=120.0 + i * 10,
            content_duration=300.0,
            # Trust signals
            has_citations=i % 2 == 0,
            citation_count=i * 2,
            source_diversity_score=0.5 + (i * 0.1),
        )
        content_items.append(item)
    
    return content_items


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
    print("\n4. Creating sample content sources and items...")
    content_sources = []
    
    for i in range(3):
        source = ContentSource(
            id=f"source_{i}",
            name=f"Content Source {i}",
            description=f"A content source about various topics",
            created_at=datetime(2023, 1, 1),
            content_items=create_sample_content(f"source_{i}", count=10),
        )
        content_sources.append(source)
    
    total_items = sum(len(source.content_items) for source in content_sources)
    print(f"   Created {len(content_sources)} content sources with {total_items} total items")
    
    # 5. Score a single content item
    print("\n5. Scoring a single content item...")
    test_item = content_sources[0].content_items[3]
    test_source = content_sources[0]
    
    # Process and index (needed for neighborhood search)
    pipeline.process_content_batch(test_source.content_items)
    pipeline.index_content(test_source.content_items)
    
    item_score = pipeline.score_content_item(test_item, test_source)
    
    print(f"\n   Content: {test_item.title}")
    print(f"   Overall IdeaRank Score: {item_score.score:.4f}")
    print(f"   Factor Breakdown:")
    print(f"     - Uniqueness (U):  {item_score.uniqueness.score:.4f}")
    print(f"     - Cohesion (C):    {item_score.cohesion.score:.4f}")
    print(f"     - Learning (L):    {item_score.learning.score:.4f}")
    print(f"     - Quality (Q):     {item_score.quality.score:.4f}")
    print(f"     - Trust (T):       {item_score.trust.score:.4f}")
    print(f"   Passes Gates: {item_score.passes_gates}")
    
    # 6. Score a content source
    print("\n6. Scoring a content source...")
    source_score = pipeline.score_source(test_source)
    
    print(f"\n   Source: {test_source.name}")
    print(f"   Source IdeaRank Score: {source_score.score:.4f}")
    print(f"   Mean Content Score: {source_score.mean_content_score:.4f}")
    print(f"   AUL Bonus: {source_score.aul_bonus:.4f}")
    print(f"   Items in Window: {source_score.content_count}")
    print(f"   Crystallization Detected: {source_score.crystallization_detected}")
    
    # 7. Compute network scores
    print("\n7. Computing KnowledgeRank across sources...")
    kr_scores = pipeline.score_sources_with_network(content_sources)
    
    print("\n   Source Rankings:")
    print("   " + "-" * 50)
    
    # Sort by KR score
    sorted_sources = sorted(
        kr_scores.items(),
        key=lambda x: x[1].knowledge_rank,
        reverse=True
    )
    
    for source_id, kr_score in sorted_sources:
        source_name = next(source.name for source in content_sources if source.id == source_id)
        print(f"   {source_name:20} | KR: {kr_score.knowledge_rank:.4f} | "
              f"IR: {kr_score.idea_rank:.4f} | "
              f"Influence: {kr_score.influence_bonus:+.4f}")
        print(f"                        | Out edges: {len(kr_score.outgoing_influence)} | "
              f"In edges: {len(kr_score.incoming_influence)}")
    
    # 8. Export results
    print("\n8. Exporting results to dict...")
    item_dict = item_score.to_dict()
    source_dict = source_score.to_dict()
    kr_dict = {source_id: score.to_dict() for source_id, score in kr_scores.items()}
    
    print(f"   Exported {len(item_dict)} content item fields")
    print(f"   Exported {len(source_dict)} source fields")
    print(f"   Exported {len(kr_dict)} KnowledgeRank scores")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

