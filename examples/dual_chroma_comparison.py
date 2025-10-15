"""Example: Compare semantic units vs regular chunks using dual Chroma collections.

This script demonstrates:
1. Processing content into both semantic units and regular chunks
2. Storing in separate Chroma collections
3. Querying both collections in parallel
4. Comparing results to see which approach works better

Usage:
    python examples/dual_chroma_comparison.py
"""

import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run dual Chroma comparison demo."""
    from idearank.models import ContentItem, ContentSource
    from idearank.providers.dual_chroma import DualChromaProvider
    
    # Initialize dual Chroma provider (local mode)
    logger.info("Initializing Dual Chroma Provider (local mode)")
    provider = DualChromaProvider(
        persist_directory="./chroma_data",
        embedding_function="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        chunk_size=8000,
        chunk_overlap=500,
    )
    
    # Example content
    example_content = """
    The Renaissance was a period of cultural rebirth in Europe that began in Italy in the 14th century.
    
    Leonardo da Vinci was one of the most influential figures of this era. He was born in 1452 in Vinci, Italy,
    and became renowned as both an artist and an inventor. His most famous paintings include the Mona Lisa
    and The Last Supper.
    
    Before the Renaissance, Leonardo was unknown. After his work gained recognition, he became one of the most
    celebrated minds of his time. This transformation was triggered by his apprenticeship with Andrea del Verrocchio,
    where he learned painting, sculpture, and mechanical arts.
    
    Michelangelo Buonarroti was another titan of the Renaissance. Born in 1475, he created masterpieces like
    the Sistine Chapel ceiling and the sculpture of David. His work emphasized human anatomy and emotion in
    unprecedented ways.
    
    The Medici family played a crucial role in supporting Renaissance artists. They were wealthy bankers in
    Florence who became patrons of the arts, funding artists like Botticelli, Michelangelo, and Leonardo.
    Their patronage transformed Florence from a commercial center into the cultural heart of the Renaissance.
    
    Scientific thinking evolved dramatically during this period. Before the Renaissance, knowledge was largely
    based on ancient texts and religious doctrine. After the Renaissance, empirical observation and experimentation
    became the foundation of knowledge. This shift was driven by figures like Galileo Galilei, who used the
    telescope to observe celestial bodies and challenge prevailing cosmological views.
    """
    
    # Create sample content item
    content_source = ContentSource(
        id="renaissance_source",
        name="Renaissance History",
        content_items=[],
    )
    
    content_item = ContentItem(
        id="renaissance_overview",
        content_source_id="renaissance_source",
        title="Overview of the Renaissance Period",
        full_text=example_content,
        published_at=datetime.now(),
    )
    
    # Process and index content in BOTH collections
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Processing content into both semantic units and chunks")
    logger.info("="*80)
    
    stats = provider.process_and_index_content(content_item, mode="both")
    
    logger.info("\nProcessing Stats:")
    logger.info(f"  Content ID: {stats['content_id']}")
    logger.info(f"  Semantic Units: {stats['semantic_units_count']}")
    logger.info(f"  Document Chunks: {stats['chunks_count']}")
    logger.info(f"  Semantic Indexed: {stats['semantic_indexed']}")
    logger.info(f"  Chunks Indexed: {stats['chunks_indexed']}")
    
    # Display collection stats
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Collection Statistics")
    logger.info("="*80)
    
    collection_stats = provider.get_stats()
    logger.info(f"\nCollection Stats:")
    for key, value in collection_stats.items():
        logger.info(f"  {key}: {value}")
    
    # Test queries
    queries = [
        "Who was Leonardo da Vinci?",
        "How did the Medici family support art?",
        "What changed in scientific thinking during the Renaissance?",
        "Tell me about Michelangelo's work",
    ]
    
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Comparing Query Results")
    logger.info("="*80)
    
    for query in queries:
        logger.info(f"\n{'='*80}")
        logger.info(f"Query: {query}")
        logger.info(f"{'='*80}")
        
        # Compare both approaches
        comparison = provider.compare_queries(query, k=5)
        
        logger.info(f"\nOverlap Metrics:")
        logger.info(f"  Overlap: {comparison['overlap_count']}/5 items ({comparison['overlap_percentage']:.1f}%)")
        logger.info(f"  Unique to Semantic: {comparison['unique_to_semantic']}")
        logger.info(f"  Unique to Chunks: {comparison['unique_to_chunks']}")
        
        logger.info(f"\n{'─'*80}")
        logger.info("Top 3 Semantic Unit Results:")
        logger.info(f"{'─'*80}")
        for i, result in enumerate(comparison['semantic_results'][:3], 1):
            logger.info(f"\n{i}. [{result['metadata'].get('unit_type', 'unknown')}] (similarity: {result['similarity']:.3f})")
            logger.info(f"   {result['document'][:200]}...")
        
        logger.info(f"\n{'─'*80}")
        logger.info("Top 3 Chunk Results:")
        logger.info(f"{'─'*80}")
        for i, result in enumerate(comparison['chunk_results'][:3], 1):
            logger.info(f"\n{i}. [Chunk {result['metadata'].get('chunk_index', '?')}] (similarity: {result['similarity']:.3f})")
            logger.info(f"   {result['document'][:200]}...")
    
    # Final analysis
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS: Semantic Units vs Regular Chunks")
    logger.info("="*80)
    
    logger.info("""
    Semantic Units Approach:
    ✓ Extracts meaningful entities (actors), events, and changes
    ✓ Self-contained units that make sense independently
    ✓ Better for knowledge graph construction
    ✓ More structured and queryable
    ✓ Naturally handles token limits (units are small)
    
    Regular Chunks Approach:
    ✓ Preserves original text flow and context
    ✓ Simpler to implement (no LLM extraction needed)
    ✓ May capture nuances that semantic extraction misses
    ✓ Good for retrieving surrounding context
    ✓ Traditional RAG approach
    
    Best Practice:
    → Use BOTH in parallel for maximum coverage
    → Semantic units for structured queries
    → Chunks for contextual retrieval
    → Combine results with re-ranking
    """)
    
    logger.info("\n" + "="*80)
    logger.info("Demo Complete!")
    logger.info("="*80)
    
    logger.info(f"\nChroma data stored in: ./chroma_data")
    logger.info("You can re-run this script to query the existing data.")


if __name__ == "__main__":
    main()

