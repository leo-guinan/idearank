"""Embedding provider interface and implementations."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import logging

from idearank.models import Embedding

logger = logging.getLogger(__name__)


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


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Provider using sentence-transformers library.
    
    Runs locally, no API key needed. Great for development and testing.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with sentence-transformers model.
        
        Args:
            model_name: Name of the sentence-transformers model
                - all-MiniLM-L6-v2: Fast, 384 dims (default)
                - all-mpnet-base-v2: Better quality, 768 dims
                - paraphrase-multilingual: Multilingual, 384 dims
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self._model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    @property
    def model_name(self) -> str:
        return f"sentence-transformers-{self._model_name}"
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed(self, text: str) -> Embedding:
        """Generate embedding using sentence-transformers."""
        vector = self.model.encode(text, convert_to_numpy=True)
        return Embedding(
            vector=vector.astype(np.float32),
            model=self.model_name
        )
    
    def embed_batch(self, texts: List[str]) -> List[Embedding]:
        """Generate embeddings for batch (more efficient)."""
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        return [
            Embedding(
                vector=vec.astype(np.float32),
                model=self.model_name
            )
            for vec in vectors
        ]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Provider using OpenAI's embedding API.
    
    Requires openai package and API key.
    """
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = ""):
        """Initialize with model and API key."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )
        
        self._model = model
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
    
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
        response = self.client.embeddings.create(
            model=self._model,
            input=text,
        )
        
        vector = np.array(response.data[0].embedding, dtype=np.float32)
        return Embedding(vector=vector, model=self.model_name)
    
    def embed_batch(self, texts: List[str]) -> List[Embedding]:
        """Generate embeddings for batch with intelligent batching to respect token limits.
        
        OpenAI has a per-text limit of 8,192 tokens and a batch limit of ~50,000 tokens.
        We'll batch intelligently to stay under these limits.
        """
        from idearank.utils.chunking import estimate_tokens
        
        MAX_TOKENS_PER_TEXT = 8000  # Per-text limit (8,192 with buffer)
        MAX_TOKENS_PER_REQUEST = 45000  # Batch limit (conservative)
        
        all_embeddings = []
        current_batch = []
        current_batch_tokens = 0
        
        for i, text in enumerate(texts):
            text_tokens = estimate_tokens(text)
            
            # If this single text exceeds the per-text limit, it needs to be handled separately
            if text_tokens > MAX_TOKENS_PER_TEXT:
                # Process current batch first
                if current_batch:
                    batch_embeddings = self._embed_batch_internal(current_batch)
                    all_embeddings.extend(batch_embeddings)
                    current_batch = []
                    current_batch_tokens = 0
                
                # This text is too large - it should have been chunked earlier
                logger.error(
                    f"Text {i} has {text_tokens} tokens (>{MAX_TOKENS_PER_TEXT}). "
                    "This should have been chunked before embedding. Skipping."
                )
                # Return a zero embedding as placeholder
                all_embeddings.append(Embedding(
                    vector=np.zeros(self.dimension, dtype=np.float32),
                    model=self.model_name
                ))
                continue
            
            # Check if adding this text would exceed the limit
            if current_batch_tokens + text_tokens > MAX_TOKENS_PER_REQUEST:
                # Process current batch
                batch_embeddings = self._embed_batch_internal(current_batch)
                all_embeddings.extend(batch_embeddings)
                current_batch = [text]
                current_batch_tokens = text_tokens
            else:
                # Add to current batch
                current_batch.append(text)
                current_batch_tokens += text_tokens
        
        # Process final batch
        if current_batch:
            batch_embeddings = self._embed_batch_internal(current_batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _embed_batch_internal(self, texts: List[str]) -> List[Embedding]:
        """Internal method to call OpenAI API for a batch that fits within limits."""
        # Filter out empty/invalid texts and ensure all are strings
        valid_texts = []
        for text in texts:
            if text and isinstance(text, str) and text.strip():
                valid_texts.append(text.strip())
        
        if not valid_texts:
            # Return empty embeddings if no valid texts
            return [Embedding(vector=np.zeros(self.dimension, dtype=np.float32), model=self.model_name) for _ in texts]
        
        response = self.client.embeddings.create(
            model=self._model,
            input=valid_texts,
        )
        
        return [
            Embedding(
                vector=np.array(item.embedding, dtype=np.float32),
                model=self.model_name
            )
            for item in response.data
        ]

