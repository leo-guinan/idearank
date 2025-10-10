"""Example using Chroma Cloud for embeddings and vector search.

This example shows how to:
1. Initialize ChromaProvider with your credentials
2. Use it for both embeddings and neighborhood search
3. Score videos using Chroma backend
"""

from datetime import datetime, timedelta
import os

from idearank import IdeaRankConfig, ContentItem, ContentSource
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


def create_sample_content(source_id: str, count: int = 10) -> list[ContentItem]:
    """Create sample content items for testing."""
    content_items = []
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
        item = ContentItem(
            id=f"{source_id}_item_{i}",
            content_source_id=source_id,
            title=f"Content {i}: {topic.title()}",
            description=f"A comprehensive guide to {topic}. " * 5,
            body=f"In this content, we explore {topic} in depth. " * 20,
            published_at=base_time + timedelta(days=i * 14),
            captured_at=datetime.now(),
            # Analytics
            view_count=1000 * (i + 1),
            impression_count=5000 * (i + 1),
            watch_time_seconds=float(800 * (i + 1) * 180),
            avg_view_duration=180.0 + i * 20,
            content_duration=600.0,
            # Trust signals
            has_citations=i % 2 == 0,
            citation_count=i * 3,
            source_diversity_score=0.6 + (i * 0.04),
        )
        content_items.append(item)
    
    return content_items


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
    print("\n4. Creating sample content sources and items...")
    content_sources = []
    
    for i in range(2):
        source = ContentSource(
            id=f"ml_source_{i}",
            name=f"ML Education Source {i}",
            description=f"Educational content about machine learning",
            created_at=datetime(2023, 1, 1),
            content_items=create_sample_content(f"ml_source_{i}", count=10),
        )
        content_sources.append(source)
    
    total_items = sum(len(source.content_items) for source in content_sources)
    print(f"   Created {len(content_sources)} sources with {total_items} items")
    
    # Process content (generate embeddings)
    print("\n5. Generating embeddings with Chroma...")
    all_items = [item for source in content_sources for item in source.content_items]
    pipeline.process_content_batch(all_items)
    print(f"   ✓ Generated {total_items} embeddings")
    
    # Index content in Chroma
    print("\n6. Indexing content in Chroma Cloud...")
    pipeline.index_content(all_items)
    print(f"   ✓ Indexed {total_items} items")
    
    # Score a single content item
    print("\n7. Scoring content with Chroma-powered search...")
    test_item = content_sources[0].content_items[5]
    test_source = content_sources[0]
    
    item_score = pipeline.score_content_item(test_item, test_source)
    
    print(f"\n   Content: {test_item.title}")
    print(f"   Overall IdeaRank Score: {item_score.score:.4f}")
    print(f"\n   Factor Breakdown:")
    print(f"     - Uniqueness (U):  {item_score.uniqueness.score:.4f}")
    print(f"       → Mean similarity to global corpus: "
          f"{item_score.uniqueness.components['mean_similarity']:.4f}")
    print(f"     - Cohesion (C):    {item_score.cohesion.score:.4f}")
    print(f"     - Learning (L):    {item_score.learning.score:.4f}")
    print(f"     - Quality (Q):     {item_score.quality.score:.4f}")
    print(f"     - Trust (T):       {item_score.trust.score:.4f}")
    
    # Score all sources
    print("\n8. Scoring content sources...")
    for source in content_sources:
        source_score = pipeline.score_source(source)
        print(f"\n   {source.name}:")
        print(f"     Source Score: {source_score.score:.4f}")
        print(f"     Mean Content Score: {source_score.mean_content_score:.4f}")
        print(f"     AUL Bonus: {source_score.aul_bonus:.4f}")
        print(f"     Crystallization: {source_score.crystallization_detected}")
    
    # Compute network scores
    print("\n9. Computing KnowledgeRank across sources...")
    config.network.enabled = True
    kr_scores = pipeline.score_sources_with_network(content_sources)
    
    print("\n   Source Rankings:")
    print("   " + "-" * 60)
    
    sorted_sources = sorted(
        kr_scores.items(),
        key=lambda x: x[1].knowledge_rank,
        reverse=True
    )
    
    for source_id, kr_score in sorted_sources:
        source_name = next(source.name for source in content_sources if source.id == source_id)
        print(f"   {source_name:25} | KR: {kr_score.knowledge_rank:.4f} | "
              f"IR: {kr_score.idea_rank:.4f}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print("\nNote: Your embeddings are now stored in Chroma Cloud and can be")
    print("      queried across sessions. Clear with collection.delete() if needed.")


if __name__ == "__main__":
    main()

