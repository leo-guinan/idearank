"""Trust (T) factor: Proof that claims are grounded.

T(item,t) = λ1·1_citations + λ2·source_diversity + λ3·(1 - corrections)

Scaled to [0,1]. Ties to the broader trust machine.

Higher score = more verifiable and reliable content.
"""

from typing import Any, Optional
import logging

from idearank.factors.base import BaseFactor, FactorResult
from idearank.models import ContentItem, ContentSource
from idearank.config import TrustConfig

logger = logging.getLogger(__name__)


class TrustFactor(BaseFactor):
    """Computes trust/integrity based on citations and corrections."""
    
    def __init__(self, config: TrustConfig):
        super().__init__(config)
        self.config: TrustConfig = config
    
    @property
    def name(self) -> str:
        return "Trust"
    
    def compute(
        self, 
        content_item: ContentItem, 
        content_source: ContentSource,
        context: Optional[dict[str, Any]] = None
    ) -> FactorResult:
        """Compute trust score using entity-idea attribution analysis.
        
        Context can contain:
            - 'total_items': int, for computing correction rate across source
            - 'use_citation_parser': bool, enable enhanced parsing
            - 'validate_citations': bool, enable AI validation
        """
        context = context or {}
        
        # Try to use citation parser for enhanced trust scoring
        use_parser = context.get('use_citation_parser', True)
        
        if use_parser:
            try:
                from idearank.citation_parser import analyze_citations
                
                # Analyze citations in content
                analysis = analyze_citations(
                    text=content_item.full_text,
                    use_spacy=True,
                    validate=context.get('validate_citations', False),
                    openai_api_key=context.get('openai_api_key'),
                    max_validations=3,  # Limit API costs
                )
                
                # Use enhanced trust score
                citation_quality = min(1.0, analysis.attribution_density / 5.0)
                entity_diversity = min(1.0, analysis.unique_entities / 10.0)
                
                # Apply validation multiplier if available
                if analysis.validation_accuracy is not None:
                    citation_quality *= (0.5 + 0.5 * analysis.validation_accuracy)
                
                has_citations = citation_quality
                source_diversity = entity_diversity
                
            except Exception as e:
                logger.warning(f"Citation parser failed, using legacy scoring: {e}")
                # Fall back to legacy
                has_citations = 1.0 if content_item.has_citations else 0.0
                source_diversity = content_item.source_diversity_score
        else:
            # Legacy scoring from content metadata
            has_citations = 1.0 if content_item.has_citations else 0.0
            source_diversity = content_item.source_diversity_score
        
        # Ensure source_diversity is bounded
        source_diversity = max(0.0, min(1.0, source_diversity))
        
        # Correction penalty
        # Compute as fraction of source items that needed corrections
        total_items = context.get('total_items', len(content_source.content_items))
        if total_items > 0:
            correction_rate = sum(item.correction_count > 0 for item in content_source.content_items) / total_items
        else:
            correction_rate = 0.0
        
        correction_factor = 1.0 - correction_rate
        
        # Weighted combination
        trust = (
            self.config.lambda1 * has_citations +
            self.config.lambda2 * source_diversity +
            self.config.lambda3 * correction_factor
        )
        
        # Normalize by sum of lambdas to ensure [0, 1]
        lambda_sum = self.config.lambda1 + self.config.lambda2 + self.config.lambda3
        if lambda_sum > 0:
            trust = trust / lambda_sum
        
        trust = max(0.0, min(1.0, trust))
        
        return FactorResult(
            score=trust,
            components={
                'has_citations': has_citations,
                'citation_count': content_item.citation_count,
                'source_diversity': source_diversity,
                'correction_rate': correction_rate,
                'correction_factor': correction_factor,
            },
            metadata={
                'lambda1': self.config.lambda1,
                'lambda2': self.config.lambda2,
                'lambda3': self.config.lambda3,
            }
        )

