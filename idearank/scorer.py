"""Main IdeaRank scoring engine.

Combines all factors (U, C, L, Q, T) to produce video and channel scores.
"""

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

from idearank.models import Video, Channel
from idearank.config import IdeaRankConfig
from idearank.factors import (
    UniquenessFactor,
    CohesionFactor,
    LearningFactor,
    QualityFactor,
    TrustFactor,
    FactorResult,
)


@dataclass
class IdeaRankScore:
    """Result of IdeaRank computation for a single video."""
    
    video_id: str
    score: float  # Final IR(v,t)
    
    # Individual factor scores
    uniqueness: FactorResult
    cohesion: FactorResult
    learning: FactorResult
    quality: FactorResult
    trust: FactorResult
    
    # Metadata
    weights_used: dict[str, float]
    passes_gates: bool  # Whether U and L meet minimum thresholds
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'video_id': self.video_id,
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
    """Computes IdeaRank scores for videos.
    
    IR(v,t) = U^w_U · C^w_C · L^w_L · Q^w_Q · T^w_T
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
    
    def score_video(
        self,
        video: Video,
        channel: Channel,
        context: Optional[dict[str, Any]] = None,
    ) -> IdeaRankScore:
        """Compute IdeaRank score for a video.
        
        Args:
            video: The video to score
            channel: The channel containing the video
            context: Context dict containing neighborhoods, analytics, etc.
            
        Returns:
            IdeaRankScore with final score and factor breakdowns
        """
        context = context or {}
        
        # Compute each factor
        uniqueness = self.uniqueness_factor.compute(video, channel, context)
        cohesion = self.cohesion_factor.compute(video, channel, context)
        learning = self.learning_factor.compute(video, channel, context)
        quality = self.quality_factor.compute(video, channel, context)
        trust = self.trust_factor.compute(video, channel, context)
        
        # Check gates
        passes_u_gate = uniqueness.score >= self.config.uniqueness.min_threshold
        passes_l_gate = learning.score >= self.config.learning.min_threshold
        passes_gates = passes_u_gate and passes_l_gate
        
        # Combine with multiplicative weights: IR = U^w_U · C^w_C · L^w_L · Q^w_Q · T^w_T
        weights = self.config.weights
        
        # Use geometric mean (product of powered factors)
        ir_score = (
            (uniqueness.score ** weights.uniqueness) *
            (cohesion.score ** weights.cohesion) *
            (learning.score ** weights.learning) *
            (quality.score ** weights.quality) *
            (trust.score ** weights.trust)
        )
        
        # If gates are not passed, apply penalty (optional - for top-tier filtering)
        if not passes_gates:
            # Could optionally reduce score here, or just mark it
            # For now, just mark and let downstream decide
            pass
        
        return IdeaRankScore(
            video_id=video.id,
            score=ir_score,
            uniqueness=uniqueness,
            cohesion=cohesion,
            learning=learning,
            quality=quality,
            trust=trust,
            weights_used={
                'uniqueness': weights.uniqueness,
                'cohesion': weights.cohesion,
                'learning': weights.learning,
                'quality': weights.quality,
                'trust': weights.trust,
            },
            passes_gates=passes_gates,
        )


@dataclass
class ChannelRankScore:
    """Result of channel-level IdeaRank computation."""
    
    channel_id: str
    score: float  # IR_S(t)
    
    # Components
    mean_video_score: float
    aul_bonus: float  # Area Under Learning bonus
    
    # Metadata
    video_count: int
    window_days: int
    crystallization_detected: bool
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'channel_id': self.channel_id,
            'score': self.score,
            'mean_video_score': self.mean_video_score,
            'aul_bonus': self.aul_bonus,
            'video_count': self.video_count,
            'window_days': self.window_days,
            'crystallization_detected': self.crystallization_detected,
        }


class ChannelScorer:
    """Computes channel-level IdeaRank scores.
    
    IR_S(t) = mean(IR(v,t)) + η·AUL(t)
    
    with anti-crystallization penalties.
    """
    
    def __init__(self, config: IdeaRankConfig):
        """Initialize channel scorer."""
        self.config = config
        self.video_scorer = IdeaRankScorer(config)
    
    def score_channel(
        self,
        channel: Channel,
        end_time: Optional[Any] = None,  # datetime
        video_scores: Optional[dict[str, IdeaRankScore]] = None,
    ) -> ChannelRankScore:
        """Compute channel-level IdeaRank score.
        
        Args:
            channel: The channel to score
            end_time: End of evaluation window (defaults to most recent video)
            video_scores: Pre-computed video scores (optional, will compute if not provided)
            
        Returns:
            ChannelRankScore with aggregate metrics
        """
        # Determine time window
        if end_time is None:
            if channel.videos:
                end_time = max(v.published_at for v in channel.videos)
            else:
                raise ValueError("Channel has no videos and no end_time provided")
        
        # Get videos in window
        videos_in_window = channel.get_videos_in_window(
            end_time,
            window_days=self.config.channel.window_days
        )
        
        if not videos_in_window:
            # No videos in window - return neutral score
            return ChannelRankScore(
                channel_id=channel.id,
                score=0.0,
                mean_video_score=0.0,
                aul_bonus=0.0,
                video_count=0,
                window_days=self.config.channel.window_days,
                crystallization_detected=False,
            )
        
        # Compute or retrieve video scores
        if video_scores is None:
            video_scores = {}
            for video in videos_in_window:
                # This would need context in practice
                # For now, placeholder
                score = self.video_scorer.score_video(video, channel, {})
                video_scores[video.id] = score
        
        # Calculate mean video score
        scores = [video_scores[v.id].score for v in videos_in_window if v.id in video_scores]
        mean_score = float(np.mean(scores)) if scores else 0.0
        
        # Calculate Area Under Learning (AUL)
        aul = self._compute_aul(videos_in_window, video_scores)
        
        # Check for crystallization
        crystallization_detected = self._detect_crystallization(videos_in_window, video_scores)
        
        # Compute final channel score
        channel_score = mean_score + self.config.channel.aul_bonus_weight * aul
        
        # Apply crystallization penalty if detected
        if crystallization_detected:
            channel_score *= self.config.channel.crystallization_decay
        
        return ChannelRankScore(
            channel_id=channel.id,
            score=channel_score,
            mean_video_score=mean_score,
            aul_bonus=aul,
            video_count=len(videos_in_window),
            window_days=self.config.channel.window_days,
            crystallization_detected=crystallization_detected,
        )
    
    def _compute_aul(
        self,
        videos: list[Video],
        video_scores: dict[str, IdeaRankScore],
    ) -> float:
        """Compute Area Under Learning.
        
        AUL = Σ max(0, L(v,t) - mean(L))
        
        Rewards consistent positive learning progression.
        """
        learning_scores = [
            video_scores[v.id].learning.score
            for v in videos
            if v.id in video_scores
        ]
        
        if not learning_scores:
            return 0.0
        
        mean_learning = float(np.mean(learning_scores))
        
        # Sum positive deviations from mean
        aul = sum(max(0.0, score - mean_learning) for score in learning_scores)
        
        return float(aul)
    
    def _detect_crystallization(
        self,
        videos: list[Video],
        video_scores: dict[str, IdeaRankScore],
    ) -> bool:
        """Detect if channel has crystallized (stopped learning).
        
        Returns True if variance in Learning scores is below floor
        for the specified number of weeks.
        """
        learning_scores = [
            video_scores[v.id].learning.score
            for v in videos
            if v.id in video_scores
        ]
        
        if len(learning_scores) < 3:
            return False  # Not enough data
        
        variance = float(np.var(learning_scores))
        
        return variance < self.config.channel.crystallization_variance_floor

