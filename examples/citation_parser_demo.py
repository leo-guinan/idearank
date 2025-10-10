"""Demonstration of the entity-idea citation parser.

Shows how the Trust factor now rewards proper attribution of ideas to people/institutions.
"""

import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

# Sample text with various attribution styles
SAMPLE_TEXT = """
According to Daniel Kahneman, human decision-making is heavily influenced by 
cognitive biases. His work on prospect theory showed that people value losses 
more than equivalent gains.

Darwin's theory of evolution by natural selection revolutionized biology. As 
Darwin argued, species evolve through differential reproductive success.

MIT researchers recently found that large language models exhibit emergent 
capabilities at scale. Stanford's AI lab has also demonstrated similar findings.

Nassim Taleb's concept of antifragility suggests that some systems benefit from 
volatility. This builds on earlier work by Mandelbrot on fractal geometry and 
power law distributions.

(Smith, 2024) proposed a new framework for understanding market dynamics. This 
challenges conventional economic theory from institutions like the Chicago School.
"""


def main():
    """Demonstrate citation parsing."""
    
    print("=" * 70)
    print("Entity-Idea Citation Parser Demo")
    print("=" * 70)
    
    # Try with spaCy first
    try:
        from idearank.citation_parser import (
            EntityIdeaCitationParser,
            analyze_citations,
            get_attribution_summary
        )
        
        print("\n[1/3] Parsing with spaCy NER...")
        parser = EntityIdeaCitationParser(use_spacy=True)
        analysis = parser.parse(SAMPLE_TEXT)
        
        print(f"\n✓ Found {analysis.total_attributions} entity-idea attributions")
        print(f"  Unique entities: {analysis.unique_entities}")
        print(f"  Institutions: {analysis.unique_institutions}")
        print(f"  Density: {analysis.attribution_density:.2f} per 1000 words")
        print(f"  Trust Score: {analysis.trust_score:.4f}")
        
        print("\n[2/3] Citation Details:")
        print("-" * 70)
        
        for i, citation in enumerate(analysis.entity_citations[:10], 1):
            print(f"\n{i}. {citation.entity_name} ({citation.entity_type})")
            print(f"   Idea: {citation.idea_snippet[:80]}...")
            print(f"   Confidence: {citation.confidence:.2f}")
        
        print("\n[3/3] Attribution Summary:")
        print("-" * 70)
        
        summary = get_attribution_summary(SAMPLE_TEXT)
        
        print(f"\nEntities and their attributed ideas:")
        for entity, ideas in summary['entity_ideas'].items():
            print(f"\n  {entity}:")
            for idea in ideas[:3]:  # Show first 3 ideas
                print(f"    - {idea[:60]}...")
        
        print(f"\n✓ Trust Score: {summary['trust_score']:.4f}")
        
        print("\n" + "=" * 70)
        print("Demo Complete!")
        print("=" * 70)
        
        print("\nWhat this means for Trust (T) factor:")
        print("  - High attribution density → Higher trust")
        print("  - Diverse entities → Bonus points")
        print("  - Institutional citations → Extra bonus")
        print("  - AI validation (optional) → Accuracy multiplier")
        
        print("\nCompare to old approach:")
        print("  ❌ Old: Count URLs (0.45 if has links, else 0.3)")
        print("  ✅ New: Analyze entity-idea attributions (0.3-1.0 range)")
        
        print("\nExample scores:")
        print("  - No attributions: 0.30")
        print("  - 2 people cited: 0.55")
        print("  - 5 people + 2 institutions: 0.75")
        print("  - 10+ diverse sources + validation: 0.90+")
        
    except ImportError as e:
        print(f"\n⚠️  spaCy not installed: {e}")
        print("\nTo use entity-idea parsing:")
        print("  pip install spacy")
        print("  python -m spacy download en_core_web_sm")
        
        print("\nFalling back to regex demo...")
        
        # Demo with regex
        from idearank.citation_parser import EntityIdeaCitationParser
        
        parser = EntityIdeaCitationParser(use_spacy=False)
        analysis = parser.parse(SAMPLE_TEXT)
        
        print(f"\n✓ Found {analysis.total_attributions} attributions (regex)")
        print(f"  Trust Score: {analysis.trust_score:.4f}")


if __name__ == "__main__":
    main()

