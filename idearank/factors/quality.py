"""Quality (Q) factor: Human time spent learning, de-biased from virality.

Q(v,t) = nzscore(WTPI(v,t)) · nzscore(CR(v,t))

where nzscore = z-score normalized within topic & size bucket, clipped to [0,1].

Higher score = genuine engagement, not clickbait.
"""

from typing import Any, Optional
import numpy as np

from idearank.factors.base import BaseFactor, FactorResult
from idearank.models import Video, Channel
from idearank.config import QualityConfig


class QualityFactor(BaseFactor):
    """Computes quality based on normalized engagement metrics."""
    
    def __init__(self, config: QualityConfig):
        super().__init__(config)
        self.config: QualityConfig = config
    
    @property
    def name(self) -> str:
        return "Quality"
    
    def compute(
        self, 
        video: Video, 
        channel: Channel,
        context: Optional[dict[str, Any]] = None
    ) -> FactorResult:
        """Compute quality score.
        
        Context should contain:
            - 'wtpi_distribution': dict with 'mean' and 'std' for normalization
            - 'cr_distribution': dict with 'mean' and 'std' for normalization
            
        If not provided, will use raw metrics (not recommended).
        """
        context = context or {}
        
        # Get watch time per impression
        wtpi = video.watch_time_per_impression
        
        # Get completion rate
        cr = video.completion_rate
        
        # Normalize WTPI
        if 'wtpi_distribution' in context:
            wtpi_norm = self._normalize_zscore(
                wtpi,
                context['wtpi_distribution']['mean'],
                context['wtpi_distribution']['std']
            )
        else:
            # Fallback: use raw value clipped to [0, 1]
            # Assume max WTPI of 600 seconds (10 minutes)
            wtpi_norm = min(1.0, wtpi / 600.0)
        
        # Normalize CR
        if 'cr_distribution' in context:
            cr_norm = self._normalize_zscore(
                cr,
                context['cr_distribution']['mean'],
                context['cr_distribution']['std']
            )
        else:
            # CR is already in [0, 1]
            cr_norm = cr
        
        # Combine with weights
        quality = (
            self.config.wtpi_weight * wtpi_norm +
            self.config.cr_weight * cr_norm
        )
        
        # Clip to [0, 1]
        quality = max(0.0, min(1.0, quality))
        
        return FactorResult(
            score=quality,
            components={
                'wtpi_raw': wtpi,
                'wtpi_normalized': wtpi_norm,
                'cr_raw': cr,
                'cr_normalized': cr_norm,
            },
            metadata={
                'has_distribution': 'wtpi_distribution' in context,
                'view_count': video.view_count,
                'impression_count': video.impression_count,
            }
        )
    
    @staticmethod
    def _normalize_zscore(value: float, mean: float, std: float) -> float:
        """Compute z-score and clip to [0, 1] range.
        
        Maps values:
        - mean-2*std → 0
        - mean → 0.5
        - mean+2*std → 1.0
        """
        if std == 0:
            return 0.5  # No variance means everything is average
        
        z = (value - mean) / std
        
        # Map z-score to [0, 1]: z=-2 → 0, z=0 → 0.5, z=2 → 1
        normalized = (z + 2) / 4
        
        return max(0.0, min(1.0, normalized))

