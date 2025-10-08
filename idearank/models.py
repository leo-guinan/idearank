"""Core data models for IdeaRank."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np
import numpy.typing as npt


@dataclass
class Embedding:
    """Semantic embedding vector for content."""
    
    vector: npt.NDArray[np.float32]
    model: str  # e.g., "openai-text-embedding-3-small"
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def dimension(self) -> int:
        """Embedding dimensionality."""
        return len(self.vector)
    
    def cosine_similarity(self, other: "Embedding") -> float:
        """Compute cosine similarity with another embedding."""
        if self.dimension != other.dimension:
            raise ValueError(f"Dimension mismatch: {self.dimension} vs {other.dimension}")
        
        dot_product = np.dot(self.vector, other.vector)
        norm_product = np.linalg.norm(self.vector) * np.linalg.norm(other.vector)
        
        if norm_product == 0:
            return 0.0
        
        return float(dot_product / norm_product)


@dataclass
class TopicMixture:
    """Topic distribution for content (e.g., from LDA/NMF)."""
    
    distribution: npt.NDArray[np.float32]  # Probabilities over topics
    topic_model: str  # e.g., "lda-50-topics"
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def num_topics(self) -> int:
        """Number of topics in the mixture."""
        return len(self.distribution)
    
    def entropy(self) -> float:
        """Shannon entropy of the topic distribution."""
        # Avoid log(0) by filtering near-zero probabilities
        probs = self.distribution[self.distribution > 1e-10]
        if len(probs) == 0:
            return 0.0
        return float(-np.sum(probs * np.log(probs)))


@dataclass
class Video:
    """Represents a video at a specific point in time."""
    
    id: str  # Unique identifier (e.g., YouTube video ID)
    channel_id: str
    title: str
    description: str
    transcript: str
    published_at: datetime
    snapshot_time: datetime  # When this data was captured
    
    # Computed representations
    embedding: Optional[Embedding] = None
    topic_mixture: Optional[TopicMixture] = None
    
    # Analytics (for Quality factor)
    view_count: int = 0
    impression_count: int = 0
    watch_time_seconds: float = 0.0
    avg_view_duration: float = 0.0
    video_duration: float = 0.0
    
    # Trust signals
    has_citations: bool = False
    citation_count: int = 0
    source_diversity_score: float = 0.0
    correction_count: int = 0
    
    # Metadata
    chapters: list[dict] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    
    @property
    def full_text(self) -> str:
        """Combined text for embedding generation."""
        parts = [self.title, self.description, self.transcript]
        return " ".join(p for p in parts if p)
    
    @property
    def watch_time_per_impression(self) -> float:
        """WTPI metric for Quality factor."""
        if self.impression_count == 0:
            return 0.0
        return self.watch_time_seconds / self.impression_count
    
    @property
    def completion_rate(self) -> float:
        """CR metric for Quality factor."""
        if self.video_duration == 0:
            return 0.0
        return min(1.0, self.avg_view_duration / self.video_duration)


@dataclass
class Channel:
    """Represents a content channel (e.g., YouTube channel)."""
    
    id: str
    name: str
    description: str
    created_at: datetime
    
    # Video history (ordered by published_at)
    videos: list[Video] = field(default_factory=list)
    
    # Channel-level metrics
    subscriber_count: int = 0
    total_views: int = 0
    
    def get_videos_in_window(
        self, 
        end_time: datetime, 
        window_days: int = 180
    ) -> list[Video]:
        """Get videos published within a time window."""
        from datetime import timedelta
        start_time = end_time - timedelta(days=window_days)
        
        return [
            v for v in self.videos
            if start_time <= v.published_at <= end_time
        ]
    
    def get_prior_video(
        self, 
        video: Video, 
        max_lookback_days: int = 270
    ) -> Optional[Video]:
        """Get the most recent video before the given video."""
        from datetime import timedelta
        cutoff = video.published_at - timedelta(days=max_lookback_days)
        
        prior_videos = [
            v for v in self.videos
            if cutoff <= v.published_at < video.published_at
        ]
        
        if not prior_videos:
            return None
        
        return max(prior_videos, key=lambda v: v.published_at)

