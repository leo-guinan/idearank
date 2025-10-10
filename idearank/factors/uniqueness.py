"""Uniqueness (U) factor: How original is this content vs. the global corpus?

U(item,t) = 1 - (1/|N_item^inter|) * Σ cos(e_item(t), e_n)

Higher score = more semantically novel vs. other content.
"""

from typing import Any, Optional
import numpy as np

from idearank.factors.base import BaseFactor, FactorResult
from idearank.models import ContentItem, ContentSource, Embedding
from idearank.config import UniquenessConfig


class UniquenessFactor(BaseFactor):
    """Computes how unique/novel a content item is compared to global content."""
    
    def __init__(self, config: UniquenessConfig):
        super().__init__(config)
        self.config: UniquenessConfig = config
    
    @property
    def name(self) -> str:
        return "Uniqueness"
    
    def compute(
        self, 
        content_item: ContentItem, 
        content_source: ContentSource,
        context: Optional[dict[str, Any]] = None
    ) -> FactorResult:
        """Compute uniqueness score.
        
        Context should contain:
            - 'global_neighbors': list of (ContentItem, similarity) tuples from ANN search
        """
        if context is None or 'global_neighbors' not in context:
            raise ValueError(
                "Uniqueness requires 'global_neighbors' in context. "
                "Run global ANN search first."
            )
        
        global_neighbors = context['global_neighbors']
        
        if not global_neighbors:
            # No neighbors means unique by default (or embedding issue)
            return FactorResult(
                score=1.0,
                components={'mean_similarity': 0.0, 'neighbor_count': 0},
                metadata={'warning': 'No global neighbors found'}
            )
        
        # Extract similarities (assuming neighbors is list of (content_item, similarity))
        if isinstance(global_neighbors[0], tuple):
            similarities = [sim for _, sim in global_neighbors[:self.config.k_global]]
        else:
            # Fallback: compute similarities if not provided
            if content_item.embedding is None:
                raise ValueError("Content item must have embedding for Uniqueness computation")
            
            similarities = [
                content_item.embedding.cosine_similarity(neighbor.embedding)
                for neighbor in global_neighbors[:self.config.k_global]
                if neighbor.embedding is not None
            ]
        
        mean_similarity = float(np.mean(similarities)) if similarities else 0.0
        uniqueness = 1.0 - mean_similarity
        
        # Clip to [0, 1] in case of numerical issues
        uniqueness = max(0.0, min(1.0, uniqueness))
        
        return FactorResult(
            score=uniqueness,
            components={
                'mean_similarity': mean_similarity,
                'neighbor_count': len(similarities),
                'min_similarity': float(np.min(similarities)) if similarities else 0.0,
                'max_similarity': float(np.max(similarities)) if similarities else 0.0,
            },
            metadata={
                'k_global': self.config.k_global,
                'meets_threshold': uniqueness >= self.config.min_threshold,
            }
        )
    
    def passes_threshold(self, result: FactorResult) -> bool:
        """Check if uniqueness meets minimum threshold for top-tier ranking."""
        return result.score >= self.config.min_threshold

