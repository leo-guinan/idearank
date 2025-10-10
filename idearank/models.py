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
class ContentItem:
    """Represents a piece of content from any source (video, blog post, tweet, etc.)."""
    
    id: str  # Unique identifier
    content_source_id: str  # ID of the source that published this
    title: str
    description: str
    body: str  # Main content text (transcript for videos, post content for blogs, tweet text for Twitter)
    published_at: datetime
    captured_at: datetime  # When this data was captured/fetched
    
    # Computed representations
    embedding: Optional[Embedding] = None
    topic_mixture: Optional[TopicMixture] = None
    
    # Analytics (for Quality factor)
    view_count: int = 0
    impression_count: int = 0
    watch_time_seconds: float = 0.0
    avg_view_duration: float = 0.0
    content_duration: float = 0.0  # Duration in seconds (for videos/audio)
    
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
        parts = [self.title, self.description, self.body]
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
        if self.content_duration == 0:
            return 0.0
        return min(1.0, self.avg_view_duration / self.content_duration)


@dataclass
class ContentSource:
    """Represents a content source (YouTube channel, blog, Twitter account, etc.)."""
    
    id: str
    name: str
    description: str
    created_at: datetime
    
    # Content history (ordered by published_at)
    content_items: list[ContentItem] = field(default_factory=list)
    
    # Source-level metrics
    subscriber_count: int = 0
    total_views: int = 0
    
    def get_content_in_window(
        self, 
        end_time: datetime, 
        window_days: int = 180
    ) -> list[ContentItem]:
        """Get content items published within a time window."""
        from datetime import timedelta
        start_time = end_time - timedelta(days=window_days)
        
        return [
            item for item in self.content_items
            if start_time <= item.published_at <= end_time
        ]
    
    def get_prior_content(
        self, 
        content_item: ContentItem, 
        max_lookback_days: int = 270
    ) -> Optional[ContentItem]:
        """Get the most recent content item before the given one."""
        from datetime import timedelta
        cutoff = content_item.published_at - timedelta(days=max_lookback_days)
        
        prior_items = [
            item for item in self.content_items
            if cutoff <= item.published_at < content_item.published_at
        ]
        
        if not prior_items:
            return None
        
        return max(prior_items, key=lambda item: item.published_at)

