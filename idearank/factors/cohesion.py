"""Cohesion (C) factor: How well does the video fit the channel's conceptual lattice?

C(v,t) = C_topic(v,t) · (1 - contradiction_rate_S(t))

where C_topic = 1 / (1 + H(mean(topic_mixtures)))

Higher score = tighter topical focus and fewer contradictions.
"""

from typing import Any, Optional
import numpy as np

from idearank.factors.base import BaseFactor, FactorResult
from idearank.models import Video, Channel, TopicMixture
from idearank.config import CohesionConfig


class CohesionFactor(BaseFactor):
    """Computes how cohesive a video is with its channel's theme."""
    
    def __init__(self, config: CohesionConfig):
        super().__init__(config)
        self.config: CohesionConfig = config
    
    @property
    def name(self) -> str:
        return "Cohesion"
    
    def compute(
        self, 
        video: Video, 
        channel: Channel,
        context: Optional[dict[str, Any]] = None
    ) -> FactorResult:
        """Compute cohesion score.
        
        Context should contain:
            - 'intra_neighbors': list of Videos from the same channel
            - 'contradiction_rate': float (optional, defaults to 0)
        """
        if context is None or 'intra_neighbors' not in context:
            raise ValueError(
                "Cohesion requires 'intra_neighbors' in context. "
                "Run intra-channel ANN search first."
            )
        
        intra_neighbors = context['intra_neighbors']
        contradiction_rate = context.get('contradiction_rate', 0.0)
        
        # Compute topical cohesion
        c_topic = self._compute_topic_cohesion(video, intra_neighbors)
        
        # Apply contradiction penalty
        if self.config.use_contradiction_penalty:
            contradiction_penalty = 1.0 - (contradiction_rate * self.config.contradiction_weight)
            contradiction_penalty = max(0.0, min(1.0, contradiction_penalty))
        else:
            contradiction_penalty = 1.0
        
        cohesion = c_topic * contradiction_penalty
        
        return FactorResult(
            score=cohesion,
            components={
                'topic_cohesion': c_topic,
                'contradiction_penalty': contradiction_penalty,
                'contradiction_rate': contradiction_rate,
            },
            metadata={
                'k_intra': self.config.k_intra,
                'window_days': self.config.window_days,
                'neighbor_count': len(intra_neighbors),
            }
        )
    
    def _compute_topic_cohesion(
        self, 
        video: Video, 
        neighbors: list[Video]
    ) -> float:
        """Compute C_topic = 1 / (1 + H(mean_topic_mixture))."""
        
        # Collect topic mixtures from neighbors
        topic_mixtures = []
        for neighbor in neighbors[:self.config.k_intra]:
            if neighbor.topic_mixture is not None:
                topic_mixtures.append(neighbor.topic_mixture.distribution)
        
        if not topic_mixtures:
            # No topic data available - neutral cohesion
            return 0.5
        
        # Average topic distributions
        mean_mixture = np.mean(topic_mixtures, axis=0)
        
        # Compute entropy
        entropy = self._shannon_entropy(mean_mixture)
        
        # Cohesion is inverse of entropy (with smoothing)
        c_topic = 1.0 / (1.0 + entropy)
        
        return float(c_topic)
    
    @staticmethod
    def _shannon_entropy(distribution: np.ndarray) -> float:
        """Compute Shannon entropy of a probability distribution."""
        # Filter out near-zero probabilities to avoid log(0)
        probs = distribution[distribution > 1e-10]
        if len(probs) == 0:
            return 0.0
        return float(-np.sum(probs * np.log(probs)))

