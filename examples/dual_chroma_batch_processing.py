"""Example: Batch process multiple content items with dual Chroma storage.

This demonstrates processing multiple pieces of content efficiently
and comparing the two approaches at scale.

Usage:
    python examples/dual_chroma_batch_processing.py
"""

import os
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_content():
    """Create sample content items for testing."""
    from idearank.models import ContentItem, ContentSource
    
    # Sample articles about different topics
    articles = [
        {
            "id": "ai_overview",
            "title": "Introduction to Artificial Intelligence",
            "text": """
            Artificial Intelligence (AI) has transformed from a theoretical concept to a practical technology
            that affects daily life. Machine learning, a subset of AI, enables computers to learn from data
            without explicit programming.
            
            Deep learning has revolutionized AI in recent years. Before deep learning, AI systems required
            extensive hand-crafted features. After the introduction of deep neural networks, systems could
            automatically learn representations from raw data. This transformation was triggered by increased
            computational power and large datasets.
            
            OpenAI is a leading AI research organization founded in 2015. They developed GPT (Generative
            Pre-trained Transformer) models that can generate human-like text. Their mission is to ensure
            that artificial general intelligence benefits all of humanity.
            """,
        },
        {
            "id": "climate_change",
            "title": "Understanding Climate Change",
            "text": """
            Climate change refers to long-term shifts in global temperatures and weather patterns. The Earth's
            average temperature has increased by approximately 1.1°C since pre-industrial times.
            
            The Intergovernmental Panel on Climate Change (IPCC) is the UN body for assessing climate science.
            Established in 1988, it provides policymakers with regular scientific assessments on climate change,
            its implications, and potential future risks.
            
            Before industrialization, atmospheric CO2 levels were around 280 parts per million. After centuries
            of burning fossil fuels, they've reached over 410 ppm. This increase was driven by coal, oil, and
            natural gas consumption for energy and transportation.
            
            Renewable energy has emerged as a solution. Solar and wind power have transformed from expensive
            alternatives to cost-competitive energy sources. Their adoption accelerated after technological
            improvements and policy support made them economically viable.
            """,
        },
        {
            "id": "quantum_computing",
            "title": "The Quantum Computing Revolution",
            "text": """
            Quantum computing harnesses quantum mechanical phenomena to process information in fundamentally
            new ways. Unlike classical bits that are either 0 or 1, quantum bits (qubits) can exist in
            superposition states.
            
            IBM has been a pioneer in quantum computing development. They launched IBM Q Network in 2017,
            providing cloud access to quantum computers for researchers and businesses. Their roadmap aims
            to build practical quantum computers with thousands of qubits.
            
            Google achieved "quantum supremacy" in 2019. Before this milestone, quantum computers couldn't
            outperform classical computers on any task. After their Sycamore processor completed a calculation
            in 200 seconds that would take classical supercomputers thousands of years, it demonstrated
            quantum advantage. This was triggered by advances in qubit coherence and error correction.
            
            Quantum cryptography could revolutionize cybersecurity. Current encryption methods rely on the
            difficulty of factoring large numbers. Quantum computers could break these, but quantum key
            distribution offers theoretically unbreakable encryption based on quantum mechanics.
            """,
        },
    ]
    
    content_source = ContentSource(
        id="tech_articles",
        name="Technology Articles",
        content_items=[],
    )
    
    content_items = []
    for i, article in enumerate(articles):
        item = ContentItem(
            id=article["id"],
            content_source_id="tech_articles",
            title=article["title"],
            full_text=article["text"],
            published_at=datetime.now() - timedelta(days=len(articles) - i),
        )
        content_items.append(item)
        content_source.content_items.append(item)
    
    return content_source, content_items


def main():
    """Run batch processing demo."""
    from idearank.providers.dual_chroma import DualChromaProvider
    
    # Initialize provider
    logger.info("Initializing Dual Chroma Provider")
    provider = DualChromaProvider(
        persist_directory="./chroma_data_batch",
        embedding_function="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
    )
    
    # Create sample content
    logger.info("\n" + "="*80)
    logger.info("Creating Sample Content")
    logger.info("="*80)
    
    content_source, content_items = create_sample_content()
    logger.info(f"Created {len(content_items)} content items")
    
    # Process all items
    logger.info("\n" + "="*80)
    logger.info("Processing Content Items (Batch)")
    logger.info("="*80)
    
    all_stats = []
    for item in content_items:
        logger.info(f"\nProcessing: {item.title}")
        stats = provider.process_and_index_content(item, mode="both")
        all_stats.append(stats)
        logger.info(f"  → {stats['semantic_units_count']} semantic units, {stats['chunks_count']} chunks")
    
    # Show aggregate stats
    logger.info("\n" + "="*80)
    logger.info("Aggregate Statistics")
    logger.info("="*80)
    
    total_semantic = sum(s['semantic_units_count'] for s in all_stats)
    total_chunks = sum(s['chunks_count'] for s in all_stats)
    
    logger.info(f"\nTotal across {len(content_items)} items:")
    logger.info(f"  Semantic Units: {total_semantic}")
    logger.info(f"  Document Chunks: {total_chunks}")
    logger.info(f"  Avg Semantic Units per item: {total_semantic / len(content_items):.1f}")
    logger.info(f"  Avg Chunks per item: {total_chunks / len(content_items):.1f}")
    
    collection_stats = provider.get_stats()
    logger.info(f"\nCollection Stats:")
    for key, value in collection_stats.items():
        logger.info(f"  {key}: {value}")
    
    # Test cross-topic queries
    logger.info("\n" + "="*80)
    logger.info("Cross-Topic Query Comparison")
    logger.info("="*80)
    
    queries = [
        "How have technologies transformed over time?",
        "What organizations are leading innovation?",
        "Tell me about quantum computing breakthroughs",
        "What are the environmental implications of technology?",
    ]
    
    for query in queries:
        logger.info(f"\n{'─'*80}")
        logger.info(f"Query: {query}")
        logger.info(f"{'─'*80}")
        
        comparison = provider.compare_queries(query, k=5)
        
        # Show which content items were retrieved
        semantic_items = set(r['content_id'] for r in comparison['semantic_results'])
        chunk_items = set(r['content_id'] for r in comparison['chunk_results'])
        
        logger.info(f"\nContent Items Retrieved:")
        logger.info(f"  Semantic approach: {', '.join(semantic_items)}")
        logger.info(f"  Chunks approach: {', '.join(chunk_items)}")
        logger.info(f"  Overlap: {comparison['overlap_percentage']:.1f}%")
        
        # Show top result from each approach
        if comparison['semantic_results']:
            top_semantic = comparison['semantic_results'][0]
            logger.info(f"\nTop Semantic Result ({top_semantic['similarity']:.3f}):")
            logger.info(f"  Type: {top_semantic['metadata'].get('unit_type', 'unknown')}")
            logger.info(f"  {top_semantic['document'][:150]}...")
        
        if comparison['chunk_results']:
            top_chunk = comparison['chunk_results'][0]
            logger.info(f"\nTop Chunk Result ({top_chunk['similarity']:.3f}):")
            logger.info(f"  {top_chunk['document'][:150]}...")
    
    logger.info("\n" + "="*80)
    logger.info("Batch Processing Complete!")
    logger.info("="*80)
    
    logger.info(f"\nData stored in: ./chroma_data_batch")


if __name__ == "__main__":
    main()

