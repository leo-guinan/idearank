"""Learning (L) factor: Is the channel advancing ideas, not repeating them?

L(v,t) = Δ_self(v,t) · R(v,t) · S(v,t)

where:
- Δ_self: semantic step from prior video (reward bounded progress)
- R: revision quality
- S: stability gate (penalize chaotic jumps)

Higher score = meaningful forward progression.
"""

from typing import Any, Optional
import numpy as np

from idearank.factors.base import BaseFactor, FactorResult
from idearank.models import Video, Channel, Embedding
from idearank.config import LearningConfig


class LearningFactor(BaseFactor):
    """Computes how much a video advances the channel's learning frontier."""
    
    def __init__(self, config: LearningConfig):
        super().__init__(config)
        self.config: LearningConfig = config
    
    @property
    def name(self) -> str:
        return "Learning"
    
    def compute(
        self, 
        video: Video, 
        channel: Channel,
        context: Optional[dict[str, Any]] = None
    ) -> FactorResult:
        """Compute learning progression score.
        
        Context can contain:
            - 'prior_video': the most recent prior video from the same channel
            - 'recent_videos': list of recent videos for stability calculation
            - 'revision_quality': float (optional, defaults to 1.0)
        """
        context = context or {}
        
        # Get prior video (from context or channel)
        prior_video = context.get('prior_video')
        if prior_video is None:
            prior_video = channel.get_prior_video(video)
        
        # If no prior video, this is the first - give neutral learning score
        if prior_video is None:
            return FactorResult(
                score=0.5,
                components={
                    'delta_self': 0.5,
                    'revision_quality': 1.0,
                    'stability': 1.0,
                },
                metadata={'is_first_video': True}
            )
        
        # Compute semantic step
        delta_self = self._compute_delta_self(video, prior_video)
        
        # Get revision quality
        revision_quality = context.get('revision_quality', 1.0)
        revision_quality = max(0.0, min(1.0, revision_quality))
        
        # Compute stability
        recent_videos = context.get('recent_videos', [])
        if not recent_videos:
            # Get recent videos from channel
            recent_videos = channel.get_videos_in_window(
                video.published_at,
                window_days=self.config.stability_window_count * 30  # rough conversion
            )[-self.config.stability_window_count:]
        
        stability = self._compute_stability(video, recent_videos)
        
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
                'prior_video_id': prior_video.id,
                'recent_video_count': len(recent_videos),
                'meets_threshold': learning >= self.config.min_threshold,
            }
        )
    
    def _compute_delta_self(self, video: Video, prior: Video) -> float:
        """Compute semantic step from prior video.
        
        Rewards progress in the target range, penalizes too small or too large steps.
        """
        if video.embedding is None or prior.embedding is None:
            return 0.5  # Neutral if embeddings missing
        
        similarity = video.embedding.cosine_similarity(prior.embedding)
        
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
    
    def _compute_stability(self, video: Video, recent_videos: list[Video]) -> float:
        """Compute stability gate: S = exp(-σ² · Var(embeddings)).
        
        Penalizes chaotic jumps in embedding space.
        """
        if len(recent_videos) < 2:
            return 1.0  # Not enough history to judge stability
        
        # Collect embeddings
        embeddings = []
        for v in recent_videos:
            if v.embedding is not None:
                embeddings.append(v.embedding.vector)
        
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

