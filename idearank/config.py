"""Configuration management for IdeaRank hyperparameters."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class FactorWeights:
    """Weights for the five main IdeaRank factors."""
    
    uniqueness: float = 0.35  # w_U
    cohesion: float = 0.20     # w_C
    learning: float = 0.25     # w_L
    quality: float = 0.15      # w_Q
    trust: float = 0.05        # w_T
    
    def validate(self) -> None:
        """Ensure weights are non-negative (don't need to sum to 1 due to product)."""
        weights = [self.uniqueness, self.cohesion, self.learning, self.quality, self.trust]
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative")
        if all(w == 0 for w in weights):
            raise ValueError("At least one weight must be positive")


@dataclass
class UniquenessConfig:
    """Configuration for Uniqueness (U) factor."""
    
    k_global: int = 50  # Number of global neighbors to compare against
    min_threshold: float = 0.15  # Minimum U score for top-tier surfacing


@dataclass
class CohesionConfig:
    """Configuration for Cohesion (C) factor."""
    
    k_intra: int = 15  # Number of intra-channel neighbors
    window_days: int = 270  # Time window for channel context
    use_contradiction_penalty: bool = True
    contradiction_weight: float = 0.3


@dataclass
class LearningConfig:
    """Configuration for Learning (L) factor."""
    
    stability_window_count: int = 10  # Number of recent videos for variance
    stability_sigma: float = 0.5  # Penalty strength for variance
    min_threshold: float = 0.05  # Minimum L score for top-tier surfacing
    target_step_size: tuple[float, float] = (0.1, 0.4)  # Ideal semantic step range


@dataclass
class QualityConfig:
    """Configuration for Quality (Q) factor."""
    
    use_topic_normalization: bool = True
    use_size_normalization: bool = True
    wtpi_weight: float = 0.5
    cr_weight: float = 0.5


@dataclass
class TrustConfig:
    """Configuration for Trust (T) factor."""
    
    lambda1: float = 0.4  # Weight for citations presence
    lambda2: float = 0.3  # Weight for source diversity
    lambda3: float = 0.3  # Weight for correction penalty


@dataclass
class ContentSourceRankConfig:
    """Configuration for content source-level scoring."""
    
    window_days: int = 180  # Sliding window for source content
    aul_bonus_weight: float = 0.1  # η - weight for Area Under Learning
    crystallization_variance_floor: float = 0.01  # Minimum learning variance
    crystallization_weeks: int = 8  # Weeks of stagnation before decay
    crystallization_decay: float = 0.95  # Soft decay factor


@dataclass
class NetworkConfig:
    """Configuration for optional KnowledgeRank network layer."""
    
    enabled: bool = False
    damping_factor: float = 0.75  # d - similar to PageRank damping
    influence_threshold: float = 0.2  # θ - minimum cosine similarity for influence
    max_lag_days: int = 90  # Maximum time lag to consider for influence
    convergence_tolerance: float = 1e-6
    max_iterations: int = 100


@dataclass
class IdeaRankConfig:
    """Master configuration for the IdeaRank algorithm.
    
    This class consolidates all hyperparameters and can be serialized
    for experiment tracking and reproducibility.
    """
    
    # Factor weights
    weights: FactorWeights = field(default_factory=FactorWeights)
    
    # Per-factor configurations
    uniqueness: UniquenessConfig = field(default_factory=UniquenessConfig)
    cohesion: CohesionConfig = field(default_factory=CohesionConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    trust: TrustConfig = field(default_factory=TrustConfig)
    
    # Higher-level configurations
    content_source: ContentSourceRankConfig = field(default_factory=ContentSourceRankConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    
    # Global settings
    embedding_model: str = "default"
    topic_model: str = "lda-50"
    random_seed: int = 42
    
    def validate(self) -> None:
        """Validate all configuration parameters."""
        self.weights.validate()
        
        # Add more validation as needed
        if self.uniqueness.k_global <= 0:
            raise ValueError("k_global must be positive")
        if self.cohesion.k_intra <= 0:
            raise ValueError("k_intra must be positive")
    
    @classmethod
    def default(cls) -> "IdeaRankConfig":
        """Create a configuration with all default values."""
        return cls()
    
    @classmethod
    def experimental(cls) -> "IdeaRankConfig":
        """Create a configuration optimized for experimentation (faster, looser)."""
        config = cls()
        config.uniqueness.k_global = 20
        config.cohesion.k_intra = 5
        config.network.max_iterations = 50
        return config

