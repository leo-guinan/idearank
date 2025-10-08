"""Provider interfaces for pluggable components.

This module defines abstract interfaces for:
- Embedding generation
- Topic modeling
- ANN search / neighborhood retrieval
"""

from idearank.providers.embeddings import EmbeddingProvider, DummyEmbeddingProvider
from idearank.providers.topics import TopicModelProvider, DummyTopicModelProvider
from idearank.providers.neighborhoods import NeighborhoodProvider, DummyNeighborhoodProvider

__all__ = [
    "EmbeddingProvider",
    "DummyEmbeddingProvider",
    "TopicModelProvider",
    "DummyTopicModelProvider",
    "NeighborhoodProvider",
    "DummyNeighborhoodProvider",
]

