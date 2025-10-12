"""
Density Factor (D)

Measures information density and content efficiency.
Optimizes differently for human vs AI audiences:
- Human: Favor concise, high-density content (assume context)
- AI: Favor explicit, comprehensive content (provide context)
"""

from dataclasses import dataclass
from typing import Literal

from idearank.models import ContentItem, ContentSource
from idearank.factors.base import BaseFactor, FactorResult


AudienceType = Literal["human", "ai", "balanced"]


@dataclass
class DensityConfig:
    """Configuration for Density factor."""
    # Weights for different density metrics
    citation_density_weight: float = 0.25
    concept_diversity_weight: float = 0.25
    information_efficiency_weight: float = 0.25
    explicitness_weight: float = 0.25
    
    # Audience optimization
    audience: AudienceType = "balanced"  # human, ai, or balanced
    
    # Reference values for normalization
    min_words: int = 100
    ideal_citations_per_1k: float = 5.0
    ideal_unique_word_ratio: float = 0.5


class DensityFactor(BaseFactor):
    """
    Density Factor: Measures information density and audience fit.
    
    Components:
    - Citation Density: How often content references sources (trust signal)
    - Concept Diversity: Topic entropy within the piece (breadth)
    - Information Efficiency: Unique words / total words (conciseness)
    - Explicitness: Self-contained vs assumes context (audience fit)
    
    Audience Optimization:
    - human: Favor high density, low explicitness (punchy, contextual)
    - ai: Favor moderate density, high explicitness (clear, comprehensive)
    - balanced: Optimize for both audiences
    """
    
    @property
    def name(self) -> str:
        """Factor name."""
        return "Density"
    
    def __init__(self, config: DensityConfig):
        self.config = config
        
    def compute(
        self,
        content_item: ContentItem,
        content_source: ContentSource,
        context: dict = None,
    ) -> FactorResult:
        """
        Compute density score for content item.
        
        Args:
            content_item: The content to score
            content_source: Parent content source
            context: Additional context (optional)
            
        Returns:
            FactorResult with density score and components
        """
        context = context or {}
        
        # Get text
        text = content_item.body or ""
        words = text.split()
        word_count = len(words)
        
        # Skip very short content
        if word_count < self.config.min_words:
            return FactorResult(
                score=0.5,  # Neutral for very short content
                components={},
                metadata={'word_count': word_count, 'too_short': True}
            )
        
        # 1. Citation Density (from trust signals)
        citation_count = getattr(content_item, 'citation_count', 0)
        citation_density = (citation_count / (word_count / 1000.0)) / self.config.ideal_citations_per_1k
        citation_density = min(1.0, citation_density)  # Cap at 1.0
        
        # 2. Concept Diversity (topic entropy)
        concept_diversity = self._compute_concept_diversity(content_item, context)
        
        # 3. Information Efficiency (unique words ratio)
        unique_words = len(set(w.lower() for w in words))
        info_efficiency = unique_words / word_count
        info_efficiency_norm = info_efficiency / self.config.ideal_unique_word_ratio
        info_efficiency_norm = min(1.0, info_efficiency_norm)
        
        # 4. Explicitness (self-contained vs contextual)
        explicitness = self._compute_explicitness(text, words, word_count)
        
        # Combine based on audience preference
        if self.config.audience == "human":
            # Humans: High density, low explicitness
            density = (
                citation_density * self.config.citation_density_weight +
                concept_diversity * self.config.concept_diversity_weight +
                info_efficiency_norm * self.config.information_efficiency_weight +
                (1.0 - explicitness) * self.config.explicitness_weight  # Invert for humans
            )
        elif self.config.audience == "ai":
            # AI: Moderate density, high explicitness
            density = (
                citation_density * 0.5 * self.config.citation_density_weight +  # Less weight on brevity
                concept_diversity * self.config.concept_diversity_weight +
                info_efficiency_norm * 0.5 * self.config.information_efficiency_weight +
                explicitness * self.config.explicitness_weight  # Favor explicit
            )
        else:  # balanced
            # Balanced: Optimize for both
            density = (
                citation_density * self.config.citation_density_weight +
                concept_diversity * self.config.concept_diversity_weight +
                info_efficiency_norm * self.config.information_efficiency_weight +
                0.5 * self.config.explicitness_weight  # Neutral on explicitness
            )
        
        # Bound to [0, 1]
        density = max(0.0, min(1.0, density))
        
        return FactorResult(
            score=density,
            components={
                'citation_density': citation_density,
                'concept_diversity': concept_diversity,
                'information_efficiency': info_efficiency_norm,
                'explicitness': explicitness,
            },
            metadata={
                'word_count': word_count,
                'unique_words': unique_words,
                'unique_word_ratio': info_efficiency,
                'audience': self.config.audience,
            }
        )
    
    def _compute_concept_diversity(self, content_item: ContentItem, context: dict) -> float:
        """
        Compute concept diversity using topic mixture entropy.
        
        Higher entropy = more diverse concepts covered.
        """
        if content_item.topic_mixture is None:
            return 0.5  # Neutral if no topic data
        
        # Compute entropy of topic distribution
        import math
        topics = content_item.topic_mixture.distribution
        
        if len(topics) == 0 or sum(topics) == 0:
            return 0.5
        
        # Normalize
        total = sum(topics)
        probs = [t / total for t in topics if t > 0]
        
        # Shannon entropy
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        
        # Normalize by max entropy (log2 of num topics)
        max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def _compute_explicitness(self, text: str, words: list, word_count: int) -> float:
        """
        Compute how explicit/self-contained the content is.
        
        Explicit content:
        - Defines terms ("X is defined as...")
        - Provides background ("In order to understand...")
        - Uses full names/titles
        - Avoids pronouns without clear antecedents
        
        Implicit content:
        - Assumes context
        - Uses "this", "that", "it" freely
        - References without explanation
        - Dense jargon
        """
        text_lower = text.lower()
        
        # Explicit markers (definitions, explanations)
        explicit_markers = [
            'is defined as', 'refers to', 'means that', 'which is',
            'in other words', 'specifically', 'for example', 'such as',
            'this means', 'to understand', 'background:', 'context:',
            'let me explain', 'to clarify', 'in simple terms',
        ]
        explicit_count = sum(text_lower.count(marker) for marker in explicit_markers)
        
        # Implicit markers (assumes context)
        implicit_markers = ['this', 'that', 'these', 'those', 'it', 'they', 'them']
        implicit_count = sum(words.count(marker) + words.count(marker.capitalize()) 
                            for marker in implicit_markers)
        
        # Normalize by word count
        explicit_density = (explicit_count * 10) / (word_count / 1000.0)  # per 1k words, weighted
        implicit_density = implicit_count / word_count
        
        # Compute explicitness score
        # High explicit markers + low implicit pronouns = high explicitness
        explicitness = min(1.0, explicit_density * 0.5 + (1.0 - implicit_density * 10) * 0.5)
        explicitness = max(0.0, explicitness)
        
        return explicitness

