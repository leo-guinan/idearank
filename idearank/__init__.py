"""IdeaRank: A PageRank replacement for ideas, not links.

This package implements a multi-factor ranking algorithm for video content
that rewards uniqueness, cohesion, learning progression, quality, and trust.
"""

from idearank.models import Video, Channel, Embedding, TopicMixture
from idearank.config import IdeaRankConfig
from idearank.scorer import IdeaRankScorer, ChannelScorer

__version__ = "0.1.0"

__all__ = [
    "Video",
    "Channel",
    "Embedding",
    "TopicMixture",
    "IdeaRankConfig",
    "IdeaRankScorer",
    "ChannelScorer",
]

