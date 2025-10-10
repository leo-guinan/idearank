"""IdeaRank: A PageRank replacement for ideas, not links.

This package implements a multi-factor ranking algorithm for content
that rewards uniqueness, cohesion, learning progression, quality, and trust.
"""

from idearank.models import ContentItem, ContentSource, Embedding, TopicMixture
from idearank.config import IdeaRankConfig
from idearank.scorer import IdeaRankScorer, ContentSourceScorer

__version__ = "2.0.0"

__all__ = [
    "ContentItem",
    "ContentSource",
    "Embedding",
    "TopicMixture",
    "IdeaRankConfig",
    "IdeaRankScorer",
    "ContentSourceScorer",
]

