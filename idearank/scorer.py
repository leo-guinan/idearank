"""Main IdeaRank scoring engine.

Combines all factors (U, C, L, Q, T, D) to produce content item and content source scores.
"""

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

from idearank.models import ContentItem, ContentSource
from idearank.config import IdeaRankConfig
from idearank.factors import (
    UniquenessFactor,
    CohesionFactor,
    LearningFactor,
    QualityFactor,
    TrustFactor,
    FactorResult,
)
from idearank.factors.density import DensityFactor


@dataclass
class IdeaRankScore:
    """Result of IdeaRank computation for a single content item."""
    
    content_item_id: str
    score: float  # Final IR(item,t)
    
    # Individual factor scores
    uniqueness: FactorResult
    cohesion: FactorResult
    learning: FactorResult
    quality: FactorResult
    trust: FactorResult
    density: FactorResult
    
    # Metadata
    weights_used: dict[str, float]
    passes_gates: bool  # Whether U and L meet minimum thresholds
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'content_item_id': self.content_item_id,
            'score': self.score,
            'factors': {
                'uniqueness': {
                    'score': self.uniqueness.score,
                    'components': self.uniqueness.components,
                },
                'cohesion': {
                    'score': self.cohesion.score,
                    'components': self.cohesion.components,
                },
                'learning': {
                    'score': self.learning.score,
                    'components': self.learning.components,
                },
                'quality': {
                    'score': self.quality.score,
                    'components': self.quality.components,
                },
                'trust': {
                    'score': self.trust.score,
                    'components': self.trust.components,
                },
            },
            'weights': self.weights_used,
            'passes_gates': self.passes_gates,
        }


class IdeaRankScorer:
    """Computes IdeaRank scores for content items.
    
    IR(item,t) = U^w_U · C^w_C · L^w_L · Q^w_Q · T^w_T · D^w_D
    """
    
    def __init__(self, config: IdeaRankConfig):
        """Initialize scorer with configuration."""
        self.config = config
        config.validate()
        
        # Initialize factor modules
        self.uniqueness_factor = UniquenessFactor(config.uniqueness)
        self.cohesion_factor = CohesionFactor(config.cohesion)
        self.learning_factor = LearningFactor(config.learning)
        self.quality_factor = QualityFactor(config.quality)
        self.trust_factor = TrustFactor(config.trust)
        self.density_factor = DensityFactor(config.density)
    
    def score_content(
        self,
        content_item: ContentItem,
        content_source: ContentSource,
        context: Optional[dict[str, Any]] = None,
    ) -> IdeaRankScore:
        """Compute IdeaRank score for a content item.
        
        Args:
            content_item: The content item to score
            content_source: The source containing the content item
            context: Context dict containing neighborhoods, analytics, etc.
            
        Returns:
            IdeaRankScore with final score and factor breakdowns
        """
        context = context or {}
        
        # Compute each factor
        uniqueness = self.uniqueness_factor.compute(content_item, content_source, context)
        cohesion = self.cohesion_factor.compute(content_item, content_source, context)
        learning = self.learning_factor.compute(content_item, content_source, context)
        quality = self.quality_factor.compute(content_item, content_source, context)
        trust = self.trust_factor.compute(content_item, content_source, context)
        density = self.density_factor.compute(content_item, content_source, context)
        
        # Check gates
        passes_u_gate = uniqueness.score >= self.config.uniqueness.min_threshold
        passes_l_gate = learning.score >= self.config.learning.min_threshold
        passes_gates = passes_u_gate and passes_l_gate
        
        # Combine with multiplicative weights: IR = U^w_U · C^w_C · L^w_L · Q^w_Q · T^w_T · D^w_D
        weights = self.config.weights
        
        # Use geometric mean (product of powered factors)
        ir_score = (
            (uniqueness.score ** weights.uniqueness) *
            (cohesion.score ** weights.cohesion) *
            (learning.score ** weights.learning) *
            (quality.score ** weights.quality) *
            (trust.score ** weights.trust) *
            (density.score ** weights.density)
        )
        
        # If gates are not passed, apply penalty (optional - for top-tier filtering)
        if not passes_gates:
            # Could optionally reduce score here, or just mark it
            # For now, just mark and let downstream decide
            pass
        
        return IdeaRankScore(
            content_item_id=content_item.id,
            score=ir_score,
            uniqueness=uniqueness,
            cohesion=cohesion,
            learning=learning,
            quality=quality,
            trust=trust,
            density=density,
            weights_used={
                'uniqueness': weights.uniqueness,
                'cohesion': weights.cohesion,
                'learning': weights.learning,
                'quality': weights.quality,
                'trust': weights.trust,
                'density': weights.density,
            },
            passes_gates=passes_gates,
        )


@dataclass
class ContentSourceRankScore:
    """Result of content source-level IdeaRank computation."""
    
    content_source_id: str
    score: float  # IR_S(t)
    
    # Components
    mean_content_score: float
    aul_bonus: float  # Area Under Learning bonus
    
    # Metadata
    content_count: int
    window_days: int
    crystallization_detected: bool
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'content_source_id': self.content_source_id,
            'score': self.score,
            'mean_content_score': self.mean_content_score,
            'aul_bonus': self.aul_bonus,
            'content_count': self.content_count,
            'window_days': self.window_days,
            'crystallization_detected': self.crystallization_detected,
        }


class ContentSourceScorer:
    """Computes content source-level IdeaRank scores.
    
    IR_S(t) = mean(IR(item,t)) + η·AUL(t)
    
    with anti-crystallization penalties.
    """
    
    def __init__(self, config: IdeaRankConfig):
        """Initialize content source scorer."""
        self.config = config
        self.content_scorer = IdeaRankScorer(config)
    
    def score_source(
        self,
        content_source: ContentSource,
        end_time: Optional[Any] = None,  # datetime
        content_scores: Optional[dict[str, IdeaRankScore]] = None,
    ) -> ContentSourceRankScore:
        """Compute content source-level IdeaRank score.
        
        Args:
            content_source: The content source to score
            end_time: End of evaluation window (defaults to most recent content)
            content_scores: Pre-computed content scores (optional, will compute if not provided)
            
        Returns:
            ContentSourceRankScore with aggregate metrics
        """
        # Determine time window
        if end_time is None:
            if content_source.content_items:
                # Get the maximum published_at datetime from content items
                published_dates = [item.published_at for item in content_source.content_items if item.published_at]
                if not published_dates:
                    raise ValueError("Content source has no valid published_at dates")
                end_time = max(published_dates)
            else:
                raise ValueError("Content source has no items and no end_time provided")
        
        # Get content in window
        items_in_window = content_source.get_content_in_window(
            end_time,
            window_days=self.config.content_source.window_days
        )
        
        if not items_in_window:
            # No content in window - return neutral score
            return ContentSourceRankScore(
                content_source_id=content_source.id,
                score=0.0,
                mean_content_score=0.0,
                aul_bonus=0.0,
                content_count=0,
                window_days=self.config.content_source.window_days,
                crystallization_detected=False,
            )
        
        # Compute or retrieve content scores
        if content_scores is None:
            content_scores = {}
            for item in items_in_window:
                # This would need context in practice
                # For now, placeholder
                score = self.content_scorer.score_content(item, content_source, {})
                content_scores[item.id] = score
        
        # Calculate mean content score
        scores = [content_scores[item.id].score for item in items_in_window if item.id in content_scores]
        mean_score = float(np.mean(scores)) if scores else 0.0
        
        # Calculate Area Under Learning (AUL)
        aul = self._compute_aul(items_in_window, content_scores)
        
        # Check for crystallization
        crystallization_detected = self._detect_crystallization(items_in_window, content_scores)
        
        # Compute final source score
        source_score = mean_score + self.config.content_source.aul_bonus_weight * aul
        
        # Apply crystallization penalty if detected
        if crystallization_detected:
            source_score *= self.config.content_source.crystallization_decay
        
        return ContentSourceRankScore(
            content_source_id=content_source.id,
            score=source_score,
            mean_content_score=mean_score,
            aul_bonus=aul,
            content_count=len(items_in_window),
            window_days=self.config.content_source.window_days,
            crystallization_detected=crystallization_detected,
        )
    
    def _compute_aul(
        self,
        content_items: list[ContentItem],
        content_scores: dict[str, IdeaRankScore],
    ) -> float:
        """Compute Area Under Learning.
        
        AUL = Σ max(0, L(item,t) - mean(L))
        
        Rewards consistent positive learning progression.
        """
        learning_scores = [
            content_scores[item.id].learning.score
            for item in content_items
            if item.id in content_scores
        ]
        
        if not learning_scores:
            return 0.0
        
        mean_learning = float(np.mean(learning_scores))
        
        # Sum positive deviations from mean
        aul = sum(max(0.0, score - mean_learning) for score in learning_scores)
        
        return float(aul)
    
    def _detect_crystallization(
        self,
        content_items: list[ContentItem],
        content_scores: dict[str, IdeaRankScore],
    ) -> bool:
        """Detect if source has crystallized (stopped learning).
        
        Returns True if variance in Learning scores is below floor
        for the specified number of weeks.
        """
        learning_scores = [
            content_scores[item.id].learning.score
            for item in content_items
            if item.id in content_scores
        ]
        
        if len(learning_scores) < 3:
            return False  # Not enough data
        
        variance = float(np.var(learning_scores))
        
        return variance < self.config.content_source.crystallization_variance_floor

