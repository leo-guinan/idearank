"""Provider interfaces for pluggable components.

This module defines abstract interfaces for:
- Embedding generation
- Topic modeling
- ANN search / neighborhood retrieval
"""

from idearank.providers.embeddings import EmbeddingProvider, DummyEmbeddingProvider
from idearank.providers.topics import TopicModelProvider, DummyTopicModelProvider
from idearank.providers.neighborhoods import NeighborhoodProvider, DummyNeighborhoodProvider

# Optional Chroma providers (only available if chromadb is installed)
try:
    from idearank.providers.chroma import (
        ChromaEmbeddingProvider,
        ChromaNeighborhoodProvider,
        ChromaProvider,
    )
    from idearank.providers.dual_chroma import DualChromaProvider
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    ChromaEmbeddingProvider = None
    ChromaNeighborhoodProvider = None
    ChromaProvider = None
    DualChromaProvider = None

__all__ = [
    "EmbeddingProvider",
    "DummyEmbeddingProvider",
    "TopicModelProvider",
    "DummyTopicModelProvider",
    "NeighborhoodProvider",
    "DummyNeighborhoodProvider",
    "ChromaEmbeddingProvider",
    "ChromaNeighborhoodProvider",
    "ChromaProvider",
    "DualChromaProvider",
    "CHROMA_AVAILABLE",
]

