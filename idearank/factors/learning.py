"""Learning (L) factor: Is the source advancing ideas, not repeating them?

L(item,t) = Δ_self(item,t) · R(item,t) · S(item,t)

where:
- Δ_self: semantic step from prior content (reward bounded progress)
- R: revision quality
- S: stability gate (penalize chaotic jumps)

Higher score = meaningful forward progression.
"""

from typing import Any, Optional
import numpy as np

from idearank.factors.base import BaseFactor, FactorResult
from idearank.models import ContentItem, ContentSource, Embedding
from idearank.config import LearningConfig


class LearningFactor(BaseFactor):
    """Computes how much a content item advances the source's learning frontier."""
    
    def __init__(self, config: LearningConfig):
        super().__init__(config)
        self.config: LearningConfig = config
    
    @property
    def name(self) -> str:
        return "Learning"
    
    def compute(
        self, 
        content_item: ContentItem, 
        content_source: ContentSource,
        context: Optional[dict[str, Any]] = None
    ) -> FactorResult:
        """Compute learning progression score.
        
        Context can contain:
            - 'prior_content': the most recent prior content from the same source
            - 'recent_items': list of recent items for stability calculation
            - 'revision_quality': float (optional, defaults to 1.0)
        """
        context = context or {}
        
        # Get prior content (from context or source)
        prior_content = context.get('prior_content')
        if prior_content is None:
            prior_content = content_source.get_prior_content(content_item)
        
        # If no prior content, this is the first - give neutral learning score
        if prior_content is None:
            return FactorResult(
                score=0.5,
                components={
                    'delta_self': 0.5,
                    'revision_quality': 1.0,
                    'stability': 1.0,
                },
                metadata={'is_first_item': True}
            )
        
        # Compute semantic step
        delta_self = self._compute_delta_self(content_item, prior_content)
        
        # Get revision quality
        revision_quality = context.get('revision_quality', 1.0)
        revision_quality = max(0.0, min(1.0, revision_quality))
        
        # Compute stability
        recent_items = context.get('recent_items', [])
        if not recent_items:
            # Get recent items from source
            recent_items = content_source.get_content_in_window(
                content_item.published_at,
                window_days=self.config.stability_window_count * 30  # rough conversion
            )[-self.config.stability_window_count:]
        
        stability = self._compute_stability(content_item, recent_items)
        
        # Combine factors
        learning = delta_self * revision_quality * stability
        
        return FactorResult(
            score=learning,
            components={
                'delta_self': delta_self,
                'revision_quality': revision_quality,
                'stability': stability,
            },
            metadata={
                'prior_content_id': prior_content.id,
                'recent_item_count': len(recent_items),
                'meets_threshold': learning >= self.config.min_threshold,
            }
        )
    
    def _compute_delta_self(self, content_item: ContentItem, prior: ContentItem) -> float:
        """Compute semantic step from prior content.
        
        Rewards progress in the target range, penalizes too small or too large steps.
        """
        if content_item.embedding is None or prior.embedding is None:
            return 0.5  # Neutral if embeddings missing
        
        similarity = content_item.embedding.cosine_similarity(prior.embedding)
        
        # Convert similarity to distance
        distance = 1.0 - similarity
        
        # Reward steps in target range
        min_step, max_step = self.config.target_step_size
        
        if distance < min_step:
            # Too similar - not enough progress
            delta = distance / min_step * 0.5  # Scale to [0, 0.5]
        elif distance > max_step:
            # Too different - might be chaotic
            delta = max(0.0, 1.0 - (distance - max_step) / (1.0 - max_step) * 0.5)
        else:
            # In target range - reward proportionally
            range_width = max_step - min_step
            position = (distance - min_step) / range_width
            delta = 0.5 + position * 0.5  # Scale to [0.5, 1.0]
        
        return float(max(0.0, min(1.0, delta)))
    
    def _compute_stability(self, content_item: ContentItem, recent_items: list[ContentItem]) -> float:
        """Compute stability gate: S = exp(-σ² · Var(embeddings)).
        
        Penalizes chaotic jumps in embedding space.
        """
        if len(recent_items) < 2:
            return 1.0  # Not enough history to judge stability
        
        # Collect embeddings
        embeddings = []
        for item in recent_items:
            if item.embedding is not None:
                embeddings.append(item.embedding.vector)
        
        if len(embeddings) < 2:
            return 1.0
        
        # Compute variance across embedding dimensions
        embeddings_array = np.array(embeddings)
        variance = float(np.var(embeddings_array))
        
        # Apply exponential decay
        sigma_sq = self.config.stability_sigma ** 2
        stability = np.exp(-sigma_sq * variance)
        
        return float(min(1.0, stability))
    
    def passes_threshold(self, result: FactorResult) -> bool:
        """Check if learning meets minimum threshold."""
        return result.score >= self.config.min_threshold

