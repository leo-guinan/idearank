"""Trust (T) factor: Proof that claims are grounded.

T(v,t) = λ1·1_citations + λ2·source_diversity + λ3·(1 - corrections)

Scaled to [0,1]. Ties to the broader trust machine.

Higher score = more verifiable and reliable content.
"""

from typing import Any, Optional

from idearank.factors.base import BaseFactor, FactorResult
from idearank.models import Video, Channel
from idearank.config import TrustConfig


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
        video: Video, 
        channel: Channel,
        context: Optional[dict[str, Any]] = None
    ) -> FactorResult:
        """Compute trust score.
        
        Context can contain:
            - 'total_videos': int, for computing correction rate across channel
        """
        context = context or {}
        
        # Citation indicator (binary: 0 or 1)
        has_citations = 1.0 if video.has_citations else 0.0
        
        # Source diversity (already scaled [0, 1])
        source_diversity = video.source_diversity_score
        source_diversity = max(0.0, min(1.0, source_diversity))
        
        # Correction penalty
        # Compute as fraction of channel videos that needed corrections
        total_videos = context.get('total_videos', len(channel.videos))
        if total_videos > 0:
            correction_rate = sum(v.correction_count > 0 for v in channel.videos) / total_videos
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
                'citation_count': video.citation_count,
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

