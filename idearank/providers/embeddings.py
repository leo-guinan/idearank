"""Embedding provider interface and implementations."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from idearank.models import Embedding


class EmbeddingProvider(ABC):
    """Abstract interface for embedding generation."""
    
    @abstractmethod
    def embed(self, text: str) -> Embedding:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[Embedding]:
        """Generate embeddings for a batch of texts."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name/identifier of the embedding model."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embeddings."""
        pass


class DummyEmbeddingProvider(EmbeddingProvider):
    """Dummy provider for testing - generates random embeddings."""
    
    def __init__(self, dimension: int = 384, seed: int = 42):
        """Initialize with fixed dimension and random seed."""
        self._dimension = dimension
        self._seed = seed
        self.rng = np.random.default_rng(seed)
    
    @property
    def model_name(self) -> str:
        return f"dummy-{self._dimension}d"
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed(self, text: str) -> Embedding:
        """Generate random embedding based on text hash (deterministic)."""
        # Use text hash for reproducibility
        text_seed = hash(text) % (2**32)
        local_rng = np.random.default_rng(text_seed)
        
        vector = local_rng.normal(0, 1, self._dimension).astype(np.float32)
        # Normalize to unit vector
        vector = vector / np.linalg.norm(vector)
        
        return Embedding(vector=vector, model=self.model_name)
    
    def embed_batch(self, texts: List[str]) -> List[Embedding]:
        """Generate embeddings for batch."""
        return [self.embed(text) for text in texts]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Provider using OpenAI's embedding API.
    
    NOTE: Requires openai package and API key.
    This is a stub - implement when needed.
    """
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = ""):
        """Initialize with model and API key."""
        self._model = model
        self.api_key = api_key
        # TODO: Initialize OpenAI client
    
    @property
    def model_name(self) -> str:
        return f"openai-{self._model}"
    
    @property
    def dimension(self) -> int:
        # text-embedding-3-small: 1536, text-embedding-3-large: 3072
        if "small" in self._model:
            return 1536
        elif "large" in self._model:
            return 3072
        else:
            return 1536  # default
    
    def embed(self, text: str) -> Embedding:
        """Generate embedding using OpenAI API."""
        # TODO: Implement actual API call
        raise NotImplementedError("OpenAI provider not yet implemented")
    
    def embed_batch(self, texts: List[str]) -> List[Embedding]:
        """Generate embeddings for batch."""
        # TODO: Implement batch API call
        raise NotImplementedError("OpenAI provider not yet implemented")

